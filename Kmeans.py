__authors__ = ["1752408", "1703664"]
__group__ = "07"

import numpy as np
from utils import colors, get_color_prob


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
        Constructor of KMeans class
        """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    def _init_X(self, mX):
        if mX.dtype != np.float64:
            mX = mX.astype("float64")

        if mX.ndim != 2:
            shape = mX.shape
            mX = mX.reshape(shape[0] * shape[1], shape[2])
        self.X = mX

    def _init_options(self, options):
        defaults = {
            "km_init": "first",
            "verbose": False,
            "tolerance": 0.0,
            "opt_DEC": 0.8,
            "max_iter": 100,
            "fitting": "WCD",
        }

        if options is None:
            self.options = defaults
        else:
            self.options = {**defaults, **options}

    def _init_centroids(self):
        if self.options["km_init"].lower() == "first":
            unique_indices = np.sort(
                np.unique(self.X, axis=0, return_index=True)[1]
            )
            self.centroids = self.X[unique_indices[: self.K]]
            self.old_centroids = self.centroids.copy()
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = self.centroids.copy()

    def get_labels(self):
        distances = distance(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1)

    def get_centroids(self):
        self.old_centroids = self.centroids.copy()
        for i in range(self.K):
            x = self.X[self.labels == i, :]
            if x.size > 0:
                self.centroids[i, :] = np.mean(x, 0)

    def converges(self):
        return np.allclose(
            self.centroids, self.old_centroids,
            atol=self.options["tolerance"], rtol=0.0
        )

    def fit(self):
        self._init_centroids()
        self.get_labels()
        i = 0
        maxIterations = self.options["max_iter"]
        converged = False
        while i < maxIterations and not converged:
            self.get_centroids()
            self.get_labels()
            converged = self.converges()
            i += 1
        self.num_iter = i

    def withinClassDistance(self):
        distance_val = (
            np.sum(np.square(self.X - self.centroids[self.labels])) / self.X.shape[0]
        )
        return distance_val

    def find_bestK(self, max_K):
        optDEC = self.options["opt_DEC"]
        self.K = 2
        self.fit()
        prevWCD = self.withinClassDistance()
        foundOptimal = False
        self.K = 3
        while self.K <= max_K and not foundOptimal:
            self.fit()
            wcd = self.withinClassDistance()
            foundOptimal = (wcd / prevWCD) > optDEC
            prevWCD = wcd
            self.K += 1
        self.K = self.K - 2


def distance(X, C):
    diff = np.tile(C, (X.shape[0], 1, 1)) - np.reshape(
        np.tile(X, C.shape[0]), (X.shape[0], C.shape[0], X.shape[1])
    )
    result = np.sqrt(np.sum((diff**2), axis=2))
    return result


def get_colors(centroids):
    result = colors[np.argmax(get_color_prob(centroids), axis=1)]
    return list(result)
