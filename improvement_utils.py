from utils import rgb2gray
import time
from Kmeans import KMeans, get_colors
from KNN import KNN
import numpy as np
import matplotlib.pyplot as plt
from my_labeling import get_color_accuracy

from utils_data import (
    read_dataset,
    read_extended_dataset,
    crop_images,
    visualize_retrieval,
)

(
    train_imgs,
    train_class_labels,
    train_color_labels,
    test_imgs,
    test_class_labels,
    test_color_labels,
) = read_dataset(root_folder="./images/", gt_json="./images/gt.json")
n = 851
# train_imgs = train_imgs[:n]
# train_class_labels = train_class_labels[:n]
# train_color_labels = train_color_labels[:n]
test_imgs = test_imgs[:n]
test_class_labels = test_class_labels[:n]
test_color_labels = test_color_labels[:n]

# List with all the existent classes
uniqueShapes = list(set(list(train_class_labels) + list(test_class_labels)))

# # Load extended ground truth
# cropped_images = crop_images(imgs, upper, lower)

knn = KNN(rgb2gray(train_imgs), train_class_labels)
shape_pred = knn.predict(rgb2gray(test_imgs), 10)
shape_prc = knn.get_percentages()
shapeAverage = []
for shape in uniqueShapes:
    shapeAv = np.mean(rgb2gray(test_imgs)[shape_pred == shape], axis=0)
    if shape == "Jeans":
        shapeAv = shapeAv[15:-15, 18:-18]
    elif shape == "Dresses":
        shapeAv = shapeAv[30:-15, 20:-20]
    elif shape == "Shirts":
        shapeAv = shapeAv[26:-28, 20:-20]
    elif shape == "Shorts":
        shapeAv = shapeAv[20:-20, 18:-18]
    shapeAverage.append(shapeAv)

visualize_retrieval(shapeAverage, len(shapeAverage))


defaults = {}
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
    print("Optimal DEC value: ", DEC)
    return bestDEC
