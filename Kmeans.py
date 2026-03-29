__authors__ = ["1752407", "1703664"]
__group__ = "07"

from math import sqrt
from typing import TypedDict
import numpy as np
import numpy.typing as npt
from pandas import unique
import utils

type farray = npt.NDArray[np.float64]
type iarray = npt.NDArray[np.int64]


class Options(TypedDict):
    km_init: str
    verbose: bool
    tolerance: int
    opt_DEC: float
    max_iter: int
    fitting: str


class KMeans:

    def __init__(self, X: farray, K: int = 1, options: Options | None = None) -> None:
        """
        Constructor of KMeans class
            Args:
                K (int): Number of cluster
                options (dict): dictionary with options
        """
        self.centroids: farray
        self.old_centroids: farray
        self.labels: iarray
        self.num_iter: int = 0
        self.K: int = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, mX: farray) -> None:
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
        Args:
            X (list or np.array): list(matrix) of all pixel values
                if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                the last dimension
        """
        if mX.dtype != np.float64:
            mX = mX.astype("float64")

        if mX.ndim != 2:
            shape = mX.shape
            mX = mX.reshape(shape[0] * shape[1], shape[2])  # pyright: ignore[reportAny]
        self.X: farray = mX

    def _init_options(self, options: Options | None) -> None:
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        defaults: Options = {
            "km_init": "first",
            "verbose": False,
            "tolerance": 0,
            "opt_DEC": 0.8,
            "max_iter": 100,
            "fitting": "WCD",
        }

        if options is None:
            self.options: Options = defaults
        else:
            self.options = defaults | options

    def _init_centroids(self) -> None:
        """
        Initialization of centroids
        """
        if self.options["km_init"].lower() == "first":
            unique_indices = np.sort(
                np.unique(self.X, axis=0, return_index=True)[1]
            )  # PERF:
            self.centroids = self.X[unique_indices[: self.K]]
            self.old_centroids = self.centroids.copy()
        else:
            self.centroids = np.random.rand(
                self.K, self.X.shape[1]  # pyright: ignore[reportAny]
            )
            self.old_centroids = self.centroids.copy()

    def get_labels(self) -> None:
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        distances = distance(self.X, self.centroids)  # [N x K]
        self.labels = np.argmin(distances, axis=1)  # [N]

    def get_centroids(self) -> None:
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        sums = np.zeros((self.K, self.X.shape[1]))
        counts = np.zeros(self.K)
        for i in range(self.X.shape[0]):
            label = self.labels[i]
            sums[label] += self.X[i]
            counts[label] += 1

        self.old_centroids = self.centroids.copy()
        for i in range(self.K):
            if counts[i] > 0:
                self.centroids[i] = sums[i] / counts[i]
            # else keep old

    def converges(self, error: float = 0) -> np.bool:
        """
        Checks if there is a difference between current and old centroids
        """
        diffs = np.subtract(self.centroids, self.old_centroids)
        return np.all(diffs <= error)

    def fit(self) -> None:
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self._init_centroids()
        i: int = 0
        maxIterations = self.options["max_iter"]
        converged = False
        while i < maxIterations and not converged:
            self.get_labels()
            self.get_centroids()
            converged = self.converges()
            i += 1

    def withinClassDistance(self) -> np.float64:
        """
        returns the within class distance of the current clustering
        """

        return (
            np.sum(np.square(np.subtract(self.X, self.centroids[self.labels])))
            / self.X.shape[0]
        )

    def find_bestK(self, max_K: int) -> int:
        """
        sets the best k analysing the results up to 'max_K' clusters
        """
        optDEC = self.options["opt_DEC"]
        self.K = 2
        self.fit()
        prevWCD = self.withinClassDistance()
        k = 3
        foundOptimal = False
        while k < max_K and not foundOptimal:
            self.K = k
            self.fit()
            wcd = self.withinClassDistance()
            foundOptimal = wcd / prevWCD < optDEC
            prevWCD = wcd
            k += 1

        if foundOptimal:
            return k - 1
        else:
            return k


def distance(
    X: npt.NDArray[np.float64], C: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################

    D = np.zeros((X.shape[0], C.shape[0]))

    for x_id, x in enumerate(X):
        for c_id, c in enumerate(C):
            dist = 0
            for i in range(X.shape[1]):
                dist += (x[i] - c[i]) ** 2
            dist = sqrt(dist)
            D[x_id][c_id] = dist

    return D


def get_colors(centroids: farray) -> iarray:
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)
    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    return np.argmax(utils.get_color_prob(centroids), axis=0)
