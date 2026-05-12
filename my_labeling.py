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
            colors = colors[:len(trueColors)]
        accuracy = len(np.intersect1d(colors, trueColors)) / len(trueColors)
        result += accuracy
    return result/len(ground_truth)



def retrieval_by_color(
    imgs=[], cols=[[]], col_pct=[[]], queries=[]
):  # TODO: Return indices
    def intersect(col_row=[], row_colPct=[]):
        idx = np.intersect1d(col_row, queries, return_indices=True)
        hit_col, hit_pct = col_row[idx], row_colPct[idx]
        idx2 = np.argsort(hit_pct)
        return hit_col[idx2], hit_pct[idx2]

    matches, match_pct = np.apply_along_axis(intersect, 1, cols)
    idx = np.argwhere(matches)
    return imgs[idx][np.argsort(match_pct[idx])]


def retrieval_by_shape(
    imgs=[], shape=[[]], neigh_count=[], queries=[]
):  # TODO: Return indices
    def intersect(labels_row=[]):
        return np.flatnonzero(np.intersect1d(labels_row, queries))

    clean_idx = np.apply_along_axis(intersect, 1, shape)
    matches, match_count = imgs[clean_idx], neigh_count[clean_idx]
    return matches[np.argsort(match_count)]


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
        retrieval_by_color(imgs, col, col_pct, col_queries),
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
    n = 10
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
        "km_init": "random",
        "verbose": False,
        "tolerance": 0,
        "opt_DEC": 0.8,
        "max_iter": 100,
        "fitting": "WCD",
    }
    # Predict
    color_pred = []
    km = [KMeans(test_imgs[i], 3, defaults) for i in range(n)]
    for classifier in km:
        classifier.find_bestK(6)
        classifier.fit()
        color_pred.append(np.array(get_colors(classifier.centroids)))

    knn = KNN(rgb2gray(train_imgs), train_class_labels)
    shape_pred = knn.predict(rgb2gray(test_imgs), 10)

    max_visualize_count = 25
    match menu():
        case "qualCol":
            query_col = input("Color query: ")
            filtered_idx = retrieval_by_color(test_imgs, color_pred, [[]], query_col)
            ok = color_pred[filtered_idx] == test_color_labels[filtered_idx]
            visualize_retrieval(
                test_imgs[filtered_idx],
                max_visualize_count,
                color_pred,
                ok,
                "Color filtering",
                query_col,
            )
        case "qualShape":
            query_col = input("Color query: ")
            filtered_idx = retrieval_by_color(trueTest, color_pred, [[]], query_col)
            ok = color_pred[filtered_idx] == trueCol[filtered_idx]
            visualize_retrieval(
                trueTest[filtered_idx],
                max_visualize_count,
                color_pred,
                ok,
                "Color filtering",
                query_col,
            )
        # case "qualCombined":
        #     query_col = input("Color query: ")
        #     query_shape = input("Shape query: ")
        #     filtered_idx = retrieval_combined(
        #         trueTest, shape_pred, [[]], color_pred, [[]], query_col, query_shape
        #     )
        #     ok = shape_pred[filtered_idx] == trueShape[filtered_idx]
        #     visualize_retrieval(
        #         trueTest[filtered_idx],
        #         max_visualize_count,
        #         shape_pred,
        #         ok,
        #         "Shape filtering",
        #         query_shape,
        #     )

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
            print(f"Shape accuracy: {get_shape_accuracy(shape_pred, test_class_labels)}")
        case "quantColAcc":
            print(f"Color accuracy: {get_color_accuracy(color_pred, test_color_labels)}")


    # Visualize
