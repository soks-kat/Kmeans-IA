__authors__ = ["1752408", "1703664"]
__group__ = "07"

import numpy as np

# import math
# import operator
from scipy.spatial.distance import cdist




class KNN:
    def row_unique(self, row):
        bincounts = np.bincount(row)
        counts = bincounts[row]
        return row[np.argmax(counts)]
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels) # (a, a, b)
        self.unique_labels, self.labels_idx = np.unique(labels, return_inverse=True) # (0, 1) (0, 0, 1)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if train_data.dtype != np.float64:
            train_data = train_data.astype("float64")

        if train_data.ndim != 2:
            shape = train_data.shape
            train_data = train_data.reshape(shape[0], shape[1] * shape[2])
        self.train_data = train_data

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if test_data.dtype != np.float64:
            test_data = test_data.astype("float64")

        if test_data.ndim != 2:
            shape = test_data.shape
            test_data = test_data.reshape(shape[0], shape[1] * shape[2])
        neighbor_idx= cdist(
            test_data, self.train_data, "euclidean"
        ).argsort(axis=1)
        self.neighbors_idx = self.labels_idx[neighbor_idx[:,:k]]
        self.neighbors = self.labels[neighbor_idx[:,:k]]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        result = np.apply_along_axis(self.row_unique, 1, self.neighbors_idx)
        return self.unique_labels[result]

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
