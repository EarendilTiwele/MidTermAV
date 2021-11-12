import os
import cv2

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
