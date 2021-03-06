import os
import cv2
import dlib

import numpy as np
import pickle as pk

from tqdm import tqdm


class Labeler:
    ''' This class allows to use always the same encoding for identities (from 0 to N-1 and -1 for unknown).
    '''

    def __init__(self, dataset=None):
        self.encoding = None
        if dataset:
            # Encode identities
            self.encoding = {}

            for id in dataset['train']:
                if id not in self.encoding:
                    self.encoding[id] = len(self.encoding)

            self.encoding['unknown'] = -1

    def encode(self, y):
        if self.encoding:
            return self.encoding[y]
        else:
            raise Exception(
                "Encoder not initialized. Pass a dataset to the constructor or load a model!")

    def save(self, output_path='labels.pkl'):
        with open(output_path, 'wb') as file:
            pk.dump(self.encoding, file)

    def load(self, input_path='labels.pkl'):
        with open(input_path, 'rb') as file:
            self.encoding = pk.load(file)


def get_db_info(path):
    '''Returns a python dict where the key represents the face id and the value the list of files.

       The folder pointed by path is structured as follows:
       path
        |- id1
        |   |- file1.jpg
        |   |- file2.jpg
        |- id2
            |- file1.jpg
            |- file2.jpg
    '''

    files = os.listdir(path)

    identities = files

    db = {}
    for id in identities:
        db[id] = os.listdir(os.path.join(path, id))

    return db


def get_dataset_filelist(dataset_path):
    '''Returns a python dict where the key represents the set and the value another dictionary containing 
       as key the face id and as value the list of files.

       The folder pointed by path is structured as follows:
       path
        |- train/val/test
           |- id1
           |   |- file1.jpg
           |   |- file2.jpg
           |- unknown
               |- file1.jpg
               |- file2.jpg
    '''
    out = {}

    for set in ['train', 'val']:
        out[set] = get_db_info(os.path.join(dataset_path, set))

    return out


def load_files(root_folder, set_files, labeler: Labeler, preprocessing_function):
    '''Returns the pre-processed images and encoded labels within a pre-defined path. 
    '''

    X = []
    y = []

    for id in tqdm(set_files, desc="Loaded identities"):
        for file in set_files[id]:
            img_path = os.path.join(root_folder, id, file)
            img = cv2.imread(img_path)
            X.append(preprocessing_function(img))
            y.append(labeler.encode(id))

    return np.array(X), np.array(y)


def load_features(root_folder, dataset, labeler: Labeler, preprocessing_function):
    '''Returns the pre-processed images and encoded labels.
    '''

    X_train, y_train = load_files(os.path.join(root_folder, 'train'),
                                  dataset['train'], labeler, preprocessing_function)
    X_val, y_val = load_files(os.path.join(root_folder, 'val'),
                              dataset['val'], labeler, preprocessing_function)

    return X_train, y_train, X_val, y_val

def _getFaceBox(net, frame):
    frameOpencv2Dnn = frame.copy()
    frameHeight = frameOpencv2Dnn.shape[0]
    frameWidth = frameOpencv2Dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencv2Dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bbox = None
    max_confidence = 0.0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence:
            max_confidence = confidence
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bbox = [x1, y1, x2, y2]
    return frameOpencv2Dnn, bbox

def _pad_bb(rect, shape, padding=20):
    # Add padding to the bbox taking into account the image shape
	rect[0] = max(0,rect[0]-padding)
	rect[1] = max(0,rect[1]-padding)
	rect[2] = min(rect[2]+padding, shape[1]-1)
	rect[3] = min(rect[3]+padding, shape[0]-1)

	return rect

def align_face(bgr_img, faceNet, fa):
    frameFace, bbox = _getFaceBox(faceNet, bgr_img)
    padding = 0
    bbox = _pad_bb(bbox, frameFace.shape, padding)
    dlibRect = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])) 
    grayframe = cv2.cvtColor(frameFace, cv2.COLOR_BGR2GRAY)
    faceim = fa.align(frameFace, grayframe, dlibRect)
    return faceim

def _get_landmarks(gray, detector, predictor):
    faces = detector(gray, 1)
    landmarks = []
    if len(faces) == 0:
        return landmarks
    rect = faces[0]
    shape = predictor(gray, rect)
    for i in range(0,68):
        p = (shape.part(i).x, shape.part(i).y)
        landmarks.append(p)
    return landmarks

def calculate_landmarks_distances(gray_image, detector, predictor):
    landmarks = _get_landmarks(gray_image, detector, predictor)
    if len(landmarks) == 0:
        return np.random.rand(67*68//2)
    distances = []
    for i, lm in enumerate(landmarks[:-1]):
        for o_lm in landmarks[i+1:]:
            distances.append(np.linalg.norm(np.array(lm)-np.array(o_lm)))
    return np.stack( distances, axis=0 )
