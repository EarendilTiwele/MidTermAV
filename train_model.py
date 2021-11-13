import os
import pickle as pk
import cv2
import numpy as np
from numpy.core.fromnumeric import shape
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils import (
    get_dataset_filelist,
    load_features,
    align_face,
    calculate_landmarks_distances,
    Labeler
)

class FaceRecognition:

    def __init__(self):
        '''FaceRecognition model constructor
        '''
        param_grid = {
            "C": [1],
            "gamma": [0.005],
        }
        self.model = GridSearchCV(SVC(kernel="rbf", class_weight="balanced", probability=True), param_grid)
        self.rejection_threshold = 0.0

    def fit(self, X_train, y_train, X_val, y_val):
        '''FaceRecognition model training. The features have been already extracted.
        '''
        self.model.fit(X_train, y_train)
        # Tune the rejection threshold on the validation set
        self.tune_rejection_threshold(X_val, y_val)

    def tune_rejection_threshold(self, X_val, y_val):
        '''Tuning of the rejection threshold.
        '''
        left = 0.0
        right = 1.0
        best_rt = 1.0
        best_acc = 0.0
        for _ in range(10):
            self.rejection_treshold = left
            results = self.predict(X_val)
            right_acc = accuracy_score(y_val, results)

            self.rejection_treshold = right
            results = self.predict(X_val)
            left_acc = accuracy_score(y_val, results)

            if left_acc >= right_acc:
                if left_acc >= best_acc:
                    best_rt = left
                    best_acc = left_acc
                right = (right+left)/2
            else:  # right è il migliore
                if right_acc >= best_acc:
                    best_rt = right
                    best_acc = right_acc
                left = (left+right)/2

        self.rejection_treshold = best_rt

    def predict(self, X):
        '''Predicts the identities of a list of faces. The features have been already extracted.
           The label is a number between 0 and N-1 if the face is recognized else -1.
           X is a list of examples to predict.
        '''
        probs = self.model.predict_proba(X)
        classes = np.argmax(probs, axis=1)
        results = []
        for i, c in enumerate(classes):
            if probs[i, c] < self.rejection_treshold:
                results.append(-1)
            else:
                results.append(c)
        return np.array(results)

    def save(self, output_path='predictor.pkl'):
        '''Saves model to be delivered in the pickle format.
        '''
        with open(output_path, 'wb') as file:
            pk.dump(dict(model=self.model, th=self.rejection_treshold), file)

    def load(self, input_path='predictor.pkl'):
        '''Loads the model from a pickle file.
        '''
        with open(input_path, 'rb') as file:
            data = pk.load(file)
            self.model = data["model"]
            self.rejection_treshold = data["th"]


def preprocessing(bgr_image):
    '''Use this function to preprocess your image (e.g. face crop, alignement, equalization, filtering, etc.)
    '''
    # return bgr_image
    debug = False
    if debug == True:
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.imshow("Original",bgr_image)
    aligned_face = align_face(bgr_image)
    img_hsv = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    equalized_face = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    filtered_face = cv2.bilateralFilter(equalized_face,9,75,75)
    if debug == True:
        cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Processed Image",filtered_face)
        cv2.waitKey(0)
    return filtered_face


def feature_extraction(X, y=None, model=None):
    '''Use this function to extract features from the train and validation sets. 
       Use the model parameter to load a pre-trained feature extractor.
    '''
    arr_distances = []
    for image in tqdm(X, desc="Calculating distances between landmarks"):
        distances = calculate_landmarks_distances(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        arr_distances.append(distances)
    print("Standardizing features")
    sc = StandardScaler()
    features_std = sc.fit_transform(arr_distances)
    if model is not None:
        features_pca = model.transform(features_std)
        return features_pca, model
    n_samples = features_std.shape[0]
    print("Searching for best number of components")
    pca = PCA(n_components=(n_samples - 1))
    pca.fit(features_std)   

    explainedVariance = pca.explained_variance_ratio_*100

    cum_variance = explainedVariance[0]
    K = 1
    for k in range(1, n_samples - 1):
        cum_variance = cum_variance+explainedVariance[k]        
        if cum_variance < 99:
            K = K + 1
        else:
            break
    print(f"Using best number of components: {K}")
    pca = PCA(n_components = K)
    pca = pca.fit(features_std)
    features_pca = pca.transform(features_std)

    return features_pca, pca


if __name__ == '__main__':

    # Load the dataset
    path = 'dataset'
    dataset_files = get_dataset_filelist(path)

    # Load the encoder
    labeler = Labeler(dataset_files)

    if not os.path.exists('labels.pkl'):
        print("WARNING: A new labels file has been created. Check the working directory.")
        labeler.save()
    else:
        labeler.load()

    # Load the files and apply the preprocessing function
    print("Loading files and applying pre-processing")
    X_train, y_train, X_val, y_val = load_features(
        path, dataset_files, labeler, preprocessing)

    print("Extracting features")
    # Compute features
    # Loading a feature extraction model if already trained.
    try:
        with open('features_model.pkl','rb') as file:
            feature_extration_model = pk.load(file)
        X, _ = feature_extraction(X_train, feature_extration_model) 
    except:
        X, feature_extration_model = feature_extraction(X_train)
        # Save feature extractor
        with open('features_model.pkl', 'wb') as file:
            pk.dump(feature_extration_model, file)
    Xv, _ = feature_extraction(X_val, model=feature_extration_model)

    # Define a Face Recognition model
    model = FaceRecognition()

    # Train the model
    model.fit(X, y_train, Xv, y_val)

    # Save the model
    model.save()
