from KNN import KNN
import numpy as np

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


def retrieval_by_color(imgs=[], cols=[[]], col_pct=[[]], queries=[]):
    def intersect(col_row=[], row_colPct=[]):
        idx = np.intersect1d(col_row, queries, return_indices=True)
        hit_col, hit_pct = col_row[idx], row_colPct[idx]
        idx2 = np.argsort(hit_pct)
        return hit_col[idx2], hit_pct[idx2]

    matches, match_pct = np.apply_along_axis(intersect, 1, cols)
    idx = np.argwhere(matches)
    return imgs[idx][np.argsort(match_pct[idx])]


def retrieval_by_shape(imgs=[], shape=[[]], neigh_count=[], queries=[]):
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
):
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
    trueTest, trueCol, trueShape = (
        test_imgs[:10],
        test_color_labels[:10],
        test_class_labels[:10],
    )
    ## Predict
    color_pred = []
    for img in trueTest:
        km = KMeans(img, 1, defaults)
        km.find_bestK(7)
        km.fit()
        color_pred.append(km.centroids)

    knn = KNN(train_imgs, train_class_labels)
    shape_pred = knn.predict(trueTest, 10)

    ## Query
    query_col = input("Color query: ")
    query_shape = input("Shape query: ")
    if query_col:
        if query_shape:
            result = retrieval_combined(
                trueTest, shape_pred, [[]], color_pred, [[]], query_col, query_shape
            )
        else:
            result = retrieval_by_color(trueTest, color_pred, [[]], query_col)
    elif query_shape:
        result = retrieval_by_shape(trueTest, shape_pred, [[]], query_shape)
    else:
        print("No queries!")

    ## Visualize Color filter
    max_visualize_count = 25
    if query_col:
        ok = color_pred == trueCol
        visualize_retrieval(
            trueTest, max_visualize_count, color_pred, ok, "Color filtering", query_col
        )
    if query_shape:
        ok = shape_pred == trueShape
        visualize_retrieval(
            trueTest,
            max_visualize_count,
            shape_pred,
            ok,
            "Shape filtering",
            query_shape,
        )
