import pickle

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class EuclideanClassifier(object):
    """
    Euclidean distance classifier
    """

    def __init__(self):
        """
        Initialize classifier
        """
        self.X = []
        self.Y = []

    def fit(self, x_train, y_train):
        """
        Add encodings and labels into classifier
        :param x_train: encodings
        :param y_train: labels
        """
        if len(self.X) > 0 and len(self.Y) > 0:
            self.X = np.concatenate((self.X, x_train), axis=0)
            self.Y = self.Y + y_train
        else:
            self.X = x_train
            self.Y = y_train

    def predict(self, x_test):
        """
        Perform prediction from given encoding
        :param x_test: given encoding
        :return: dictionary with label and confidence (euclidean distance)
        """
        distances = euclidean_distances(np.expand_dims(x_test, axis=0), self.X)[0]
        idx = int(np.argmin(distances))
        max_val, min_val = 1, -1
        raw_confidence = max(1 - distances[idx], min_val)
        return {"person": self.Y[idx], "confidence": (raw_confidence - min_val) / (max_val - min_val)}

    def load(self, path):
        """
        Load classifier from pickle file
        :param path: path
        """
        database = pickle.load(open(path, "rb"))

        self.X = database["encodings"]
        self.Y = database["people"]

    def save(self, path):
        """
        Save classifier as pickle file
        :param path: path
        """

        database = {
            "encodings": self.X,
            "people": self.Y
        }

        pickle.dump(database, open(path, "wb"))
