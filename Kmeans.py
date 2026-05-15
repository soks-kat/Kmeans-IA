__authors__ = ["1752407", "1703664"]
__group__ = "07"

import numpy as np
from utils import colors, get_color_prob, oklab2rgb


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

        mX = mX[np.all(mX != 255, axis=1)]

        self.X = mX

    def _init_options(self, options):
        defaults = {
            "km_init": "first",
            "verbose": False,
            "tolerance": 0.0,
            "opt_DEC":  0.74285,
            "max_iter": 100,
            "fitting": "WCD",
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

        elif self.options["km_init"].lower() == "basic":
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

        elif self.options["km_init"].lower() == "kmeans++":
            rng = np.random.default_rng()
            self.centroids = np.zeros((self.K, 3))
            self.centroids[0] = rng.choice(self.X)
            for i in range(1, self.K):
                distances = np.min(distance(self.X, self.centroids[:i]), axis=1)
                p = distances / np.sum(distances, dtype=float)
                self.centroids[i] = rng.choice(self.X, p=p)

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

    def get_percentages(self):
        percentages = np.zeros(self.K)
        for i in range(self.K):
            x = self.X[self.labels == i, :]
            percentages[i] = x.shape[0]

        certainty = np.max(get_color_prob(np.apply_along_axis(oklab2rgb, 1, self.centroids)), axis=1)
        return percentages / self.X.shape[0]

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

    def daviesBouldiniIndex(self):
        s = np.zeros(self.K)
        for k in range(self.K):
            mask = (self.labels == k)
            s[k] = np.linalg.norm(self.X[mask] - self.centroids[k], axis=1).mean()
        diff = self.centroids[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        centroidDistance = np.linalg.norm(diff, axis=2)
        bigS = s[:, np.newaxis] + s[np.newaxis, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(bigS, centroidDistance, where=centroidDistance>0)
            ratio[centroidDistance == 0] = np.inf
            np.fill_diagonal(ratio, -np.inf)
        R_i = np.max(ratio, axis=1)
        return np.mean(R_i)



    def find_bestK(self, max_K):
        optDEC = self.options["opt_DEC"]
        if self.options["fitting"] == "WCD":
            self.K = 2
            self.fit()
            prev = self.withinClassDistance()
            foundOptimal = False
            values = [prev]
            while self.K < max_K and not foundOptimal:
                self.K += 1
                self.fit()
                current = self.withinClassDistance()
                values.append(current)
                foundOptimal = (current / prev) > optDEC
                prev = current
            self.K = self.K - 2
            return values

        elif self.options["fitting"] == "DBI":
            self.K = 2
            values = []
            while self.K < max_K:
                self.fit()
                values.append(self.daviesBouldiniIndex())
                self.K += 1
            self.K = np.argmin(values) + 2
            return values
        else: raise Exception("Fitting option invalid")


def distance(X, C):
    diff = np.tile(C, (X.shape[0], 1, 1)) - np.reshape(
        np.tile(X, C.shape[0]), (X.shape[0], C.shape[0], X.shape[1])
    )
    result = np.sqrt(np.sum((diff**2), axis=2))
    return result


def get_colors(centroids):
    result = colors[np.argmax(get_color_prob(np.apply_along_axis(oklab2rgb, 1, centroids)), axis=1)]
    return list(result)

def remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag
