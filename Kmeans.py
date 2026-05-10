__authors__ = ["1752407", "1703664"]
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
            "opt_DEC": 0.2,
            "max_iter": 100,
            "fitting": "ICD",
        }

        if options is None:
            self.options = defaults
        else:
            self.options = {**defaults, **options}

    def _init_centroids(self):
        assert type(self.options["km_init"]) is str
        if self.options["km_init"].lower() == "first":
            unique_indices = np.sort(np.unique(self.X, axis=0, return_index=True)[1])
            self.centroids = self.X[unique_indices[: self.K]]

        if self.options["km_init"].lower() == "basic":
            assert self.K <= 14, "This method can at most implement 11 colors"
            basic_colors = np.array(
                [
                    [0, 0, 0],
                    [255, 0, 0],
                    [255, 128, 0],
                    [255, 0, 128],
                    [0, 255, 0],
                    [128, 255, 0],
                    [0, 255, 128],
                    [0, 0, 255],
                    [128, 0, 255],
                    [0, 128, 255],
                    [255, 255, 0],
                    [255, 0, 255],
                    [0, 255, 255],
                    [255, 255, 255],
                ]
            )
            closest = basic_colors[
                np.argsort(np.sum(distance(basic_colors, self.X), axis=1))
            ]
            self.centroids = closest[: self.K]

        if self.options["km_init"].lower() == "kmeans++":
            rng = np.random.default_rng()
            self.centroids = np.zeros((self.K, 3))
            p = np.ones(self.X.shape[0]) / self.X.shape[0]
            for i in range(1, self.K):
                self.centroids[i] = rng.choice(self.X, p=p)
                distances = np.min(distance(self.X, self.centroids[:i]), axis=1)
                p = distances / np.sum(distances, dtype=float)

        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1]) * 255

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
            self.centroids, self.old_centroids, atol=self.options["tolerance"], rtol=0.0
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

    def interClassDistance(self):
        result = 0
        for i in range(self.K):
            matchingIdx = self.labels == i
            result += np.sum(
                np.square(
                    np.min(
                        distance(self.X[matchingIdx], self.X[~matchingIdx]),
                        axis=1,
                    )
                )
            )

        distance_val = result / self.X.shape[0]
        return distance_val

    def find_bestK(self, max_K):
        optDEC = self.options["opt_DEC"]
        if self.options["fitting"] == "WCD":
            self.K = 2
            self.fit()
            prev = self.withinClassDistance()
            foundOptimal = False
            self.K = 3
            while self.K <= max_K and not foundOptimal:
                self.fit()
                current = self.withinClassDistance()
                foundOptimal = (current / prev) > optDEC
                prev = current
                self.K += 1
            self.K = self.K - 2
        elif self.options["fitting"] == "ICD":
            self.K = 2
            self.fit()
            prev = self.interClassDistance()
            foundOptimal = False
            self.K = 3
            while self.K <= max_K and not foundOptimal:
                self.fit()
                current = self.interClassDistance()
                foundOptimal = (current / prev) < optDEC
                prev = current
                self.K += 1
            self.K = self.K - 2
        elif self.options["fitting"] == "Fisher":
            self.K = 2
            self.fit()
            prev = self.withinClassDistance() / self.interClassDistance()
            foundOptimal = False
            self.K = 3
            while self.K <= max_K and not foundOptimal:
                self.fit()
                current = self.withinClassDistance() / self.interClassDistance()
                foundOptimal = (current / prev) > optDEC
                prev = current
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
