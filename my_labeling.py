from utils import rgb2gray
import time
from Kmeans import KMeans, get_colors
from KNN import KNN
import numpy as np
import matplotlib.pyplot as plt

__authors__ = "TO_BE_FILLED"
__group__ = "TO_BE_FILLED"

from Kmeans import KMeans
from utils_data import (
    read_dataset,
    read_extended_dataset,
    crop_images,
    visualize_retrieval,
)
import utils_data


def menu():
    t = ""
    while t not in ("1", "2"):
        print("Select type of analysis to perform:")
        print(" 1. Qualitative")
        print(" 2. Quantitative")
        t = input("==> ")
    m = ""
    if t == "1":
        while m not in ("1", "2", "3"):
            print(" Select procedure:")
            print(" 1. Color Retrieval")
            print(" 2. Shape Retrieval")
            print(" 3. Combined retrieval")

            m = input("==> ")
        return ["qualCol", "qualShape", "qualCombined"][int(m) - 1]
    elif t == "2":
        while m not in ("1", "2", "3"):
            print("Select procedure: ")
            print(" 1. Kmean_statistics")
            print(" 2. Shape Accuracy")
            print(" 3. Color Accuracy")
            m = input("==> ")
        return ["quantKmeanStats", "quantShapeAcc", "quantColAcc"][int(m) - 1]
    raise Exception("Invalid Menu option")


def kmean_statistics(classifier: KMeans, Kmax):
    wcd = np.zeros(Kmax - 1)
    iterations = np.zeros(Kmax - 1)
    convTime = np.zeros(Kmax - 1)
    for i in range(2, Kmax):
        classifier.K = i
        init = time.time()
        classifier.fit()

        convTime[i - 2] = time.time() - init
        wcd[i - 2] = classifier.withinClassDistance()
        iterations[i - 2] = classifier.num_iter

    plt.subplot(131)
    plt.title("Convergence Time (ms)")
    plt.xlabel("K")
    plt.plot(range(2, Kmax + 1), convTime)

    plt.subplot(132)
    plt.title("WCD")
    plt.xlabel("K")
    plt.plot(range(2, Kmax + 1), wcd)

    plt.subplot(133)
    plt.title("Iterations")
    plt.xlabel("K")
    plt.plot(range(2, Kmax + 1), iterations)

    plt.show()


def get_shape_accuracy(shape_labels, ground_truth):
    return sum(shape_labels == ground_truth) / len(shape_labels)


def get_color_accuracy(color_labels, ground_truth):
    result = 0
    for colors, trueColors in zip(color_labels, ground_truth):
        if len(colors) > len(trueColors):
            colors = colors[: len(trueColors) + 1]
        accuracy = len(np.intersect1d(colors, trueColors)) / len(trueColors)
        result += accuracy
    return result / len(ground_truth)


def retrieval_by_color(
    imgs=[], cols=[[]], col_pct=[[]], queries=[]
):  # TODO: Return indices

    def intersect(col_row=[], row_colPct=[]):
        row, idx, temp = np.intersect1d(col_row, queries, return_indices=True)
        hit_col = col_row[idx]
        hit_pct = row_colPct[idx]
        return hit_col.size != 0, np.sum(hit_pct) * len(hit_col) / len(queries)
        # return idx2

    # matches, match_pct = np.apply_along_axis(intersect, 1, cols)
    matches = np.empty(len(cols), dtype=str)
    match_pct = np.empty(len(cols), dtype=float)
    for i, (x, y) in enumerate(zip(cols, col_pct)):
        matches[i], match_pct[i] = intersect(x, y)
    idx = np.where(matches == "T")[0]
    return idx[np.argsort(match_pct[idx])[::-1]]


def retrieval_by_shape(
    imgs=[], shapes=[[]], shape_pct=[], queries=[]
):  # TODO: Return indices
    def intersect(shape_row=[], row_shapePct=[]):
        row = np.intersect1d(shape_row, queries, assume_unique=True)
        return row.size != 0, np.sum(row_shapePct)
        # return idx2

    # matches, match_pct = np.apply_along_axis(intersect, 1, shapes)
    matches = np.empty(len(shapes), dtype=str)
    match_pct = np.empty(len(shapes), dtype=float)
    for i, (x, y) in enumerate(zip(shapes, shape_pct)):
        matches[i], match_pct[i] = intersect(x, y)
    idx = np.where(matches == "T")[0]
    return idx[np.argsort(match_pct[idx])[::-1]]


def retrieval_combined(
    imgs=[],
    shape=[[]],
    shape_pct=[[]],
    col=[[]],
    col_pct=[[]],
    col_queries=[],
    shape_queries=[],
):  # TODO: Return indices
    return retrieval_by_shape(
        imgs[retrieval_by_color(imgs, col, col_pct, col_queries)],
        shape,
        shape_pct,
        shape_queries,
    )


if __name__ == "__main__":

    # Load all the images and GT
    (
        train_imgs,
        train_class_labels,
        train_color_labels,
        test_imgs,
        test_class_labels,
        test_color_labels,
    ) = read_dataset(root_folder="./images/", gt_json="./images/gt.json")
    n = 500
    # train_imgs = train_imgs[:n]
    # train_class_labels = train_class_labels[:n]
    # train_color_labels = train_color_labels[:n]
    test_imgs = test_imgs[:n]
    test_class_labels = test_class_labels[:n]
    test_color_labels = test_color_labels[:n]

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # # Load extended ground truth
    # imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    # cropped_images = crop_images(imgs, upper, lower)

    defaults = {
    }
    # Predict
    color_pred = []
    color_prc = []
    km = [KMeans(test_imgs[i], 3, defaults) for i in range(n)]

    def findBestDEC(minDEC, maxDEC, pointCount, repetitions=3):
        accuracies = []
        bestDEC = 0
        bestAcc = 0
        for _ in range(repetitions):
            for DEC in np.linspace(minDEC, maxDEC, pointCount):
                color_pred = []
                color_prc = []
                for classifier in km:
                    classifier.options["opt_DEC"] = DEC
                    indexVals = classifier.find_bestK(7)
                    classifier.fit()
                    colors = np.array(get_colors(classifier.centroids))
                    prcs = np.array(classifier.get_percentages())
                    sorted_idx = np.argsort(prcs)[::-1]
                    colors, prcs = colors[sorted_idx], prcs[sorted_idx]
                    color_pred.append(colors)
                    color_prc.append(prcs)
                accuracy = get_color_accuracy(color_pred, test_color_labels)
                if accuracy > bestAcc:
                    bestAcc = accuracy
                    bestDEC = DEC
        for classifier in km:
            classifier.options["opt_DEC"] = bestDEC
        return bestDEC

    # DEC = findBestDEC(0.73, 0.76, 8)
    # print("Optimal DEC value: ", DEC)
    i = 0
    for classifier in km:
        indexVals = classifier.find_bestK(7)
        classifier.fit()
        colors = np.array(get_colors(classifier.centroids))
        prcs = np.array(classifier.get_percentages())
        sorted_idx = np.argsort(prcs)[::-1]
        colors, prcs = colors[sorted_idx], prcs[sorted_idx]
        color_pred.append(colors)
        color_prc.append(prcs)
        i += 1

    knn = KNN(rgb2gray(train_imgs), train_class_labels)
    shape_pred = knn.predict(rgb2gray(test_imgs), 10)
    shape_prc = knn.get_percentages()

    max_visualize_count = 25
    while True:
        match menu():
            case "qualCol":
                query_col = input("Color query: ")
                array_col = np.array(
                    [x.strip().lower().capitalize() for x in query_col.split(",")]
                )
                filtered_idx = retrieval_by_color(
                    test_imgs, color_pred, color_prc, array_col
                )

                ok = np.empty_like(filtered_idx)
                for i, trueColors in enumerate(test_color_labels[filtered_idx]):
                    ok[i] = len(np.intersect1d(array_col, trueColors)) != 0
                visualize_retrieval(
                    test_imgs[filtered_idx],
                    max_visualize_count,
                    None,
                    ok,
                    "Color filtering",
                    query_col,
                )
            case "qualShape":
                query_shape = input("Shape query: ")
                array_shape = np.array(
                    [x.strip().lower().capitalize() for x in query_shape.split(",")]
                )
                filtered_idx = retrieval_by_shape(
                    test_imgs, shape_pred, shape_prc, array_shape
                )
                ok = shape_pred[filtered_idx] == test_class_labels[filtered_idx]
                visualize_retrieval(
                    test_imgs[filtered_idx],
                    max_visualize_count,
                    None,
                    ok,
                    "Color filtering",
                    query_shape,
                )
            case "qualCombined":
                query_col = input("Color query: ")
                array_col = np.array(
                    [x.strip().lower().capitalize() for x in query_col.split(",")]
                )
                query_shape = input("Shape query: ")
                array_shape = np.array(
                    [x.strip().lower().capitalize() for x in query_shape.split(",")]
                )
                filtered_idx = retrieval_combined(
                    test_imgs,
                    shape_pred,
                    shape_prc,
                    color_pred,
                    color_prc,
                    array_col,
                    array_shape,
                )
                ok = shape_pred[filtered_idx] == test_class_labels[filtered_idx]
                visualize_retrieval(
                    test_imgs[filtered_idx],
                    max_visualize_count,
                    ok,
                    "Shape filtering",
                    query_col + " " + query_shape,
                )

            case "quantKmeanStats":
                # TODO: Select image
                t = ""
                while not t:
                    print()
                    t = input(f"Select image from 1 to {n}: ")
                    try:
                        t = int(t)
                    except:
                        print("Invalid Selection")
                        t = ""
                kmax = ""
                while not kmax:
                    print()
                    kmax = input(f"Select k: ")
                    try:
                        kmax = int(kmax)
                    except:
                        kmax = ""
                kmean_statistics(km[t - 1], kmax)
            case "quantShapeAcc":
                print(
                    f"Shape accuracy: {get_shape_accuracy(shape_pred, test_class_labels)}"
                )
            case "quantColAcc":
                print(
                    f"Color accuracy: {get_color_accuracy(color_pred, test_color_labels)}"
                )

        # Visualize
