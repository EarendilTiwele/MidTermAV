import os
import pickle as pk
import dlib
import cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from face_aligner import FaceAligner

from utils import (
    get_dataset_filelist,
    load_features,
    align_face,
    Labeler
)

class FaceRecognition:

    def __init__(self):
        '''FaceRecognition model constructor
        '''
        self.model = None
        pass

    def fit(self, X_train, y_train, X_val, y_val):
        '''FaceRecognition model training. The features have been already extracted.
        '''
        param_grid = {
            "C": [1],
            "gamma": [0.005],
        }
        clf = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid)
        self.model = clf.fit(X_train, y_train)
        # Tune the rejection threshold on the validation set
        #self.tune_rejection_threshold(X_val, y_val)

    def tune_rejection_threshold(self, X_val, y_val):
        '''Tuning of the rejection threshold.
        '''
        pass

    def predict(self, X):
        '''Predicts the identities of a list of faces. The features have been already extracted.
           The label is a number between 0 and N-1 if the face is recognized else -1.
           X is a list of examples to predict.
        '''
        return self.model.predict(X)

    def save(self, output_path='predictor.pkl'):
        '''Saves model to be delivered in the pickle format.
        '''
        file_path = "model.pkl"
        with open(file_path,'wb') as file:
               pk.dump(self.model, file)

    def load(self, input_path='predictor.pkl'):
        '''Loads the model from a pickle file.
        '''
        file_path = "model.pkl"
        with open(file_path,'rb') as file:
               self.model = pk.load(file)


def preprocessing(bgr_image):
    '''Use this function to preprocess your image (e.g. face crop, alignement, equalization, filtering, etc.)
    '''
    aligned_face = align_face(bgr_image)
    equalized_face = cv2.equalizeHist(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY))
    return equalized_face


def feature_extraction(X, y=None, model=None):
    '''Use this function to extract features from the train and validation sets. 
       Use the model parameter to load a pre-trained feature extractor.
    '''
    n_components = 12

    print("Extracting the top %d eigenfaces from %d faces" %
          (n_components, X.shape[0]))
    reshaped_X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    pca = PCA(n_components=n_components, svd_solver="randomized",
              whiten=True).fit(reshaped_X)
    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_new = pca.transform(reshaped_X)
    return X_new, model


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

    # Compute features
    # Loading a feature extraction model if already trained.
    with open('features_model.pkl','rb') as file:
        feature_extration_model = pk.load(file)

    X, _ = feature_extraction(X_train, feature_extration_model) 
    
    print("Extractin features")
    #X, feature_extration_model = feature_extraction(X_train, y_train)
    Xv, _ = feature_extraction(X_val, model=feature_extration_model)

    # Save feature extractor
    with open('features_model.pkl', 'wb') as file:
        pk.dump(feature_extration_model, file)

    # Define a Face Recognition model
    model = FaceRecognition()

    # Train the model
    model.fit(X, y_train, Xv, y_val)

    # Save the model
    model.save()
