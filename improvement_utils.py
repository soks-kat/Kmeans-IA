from utils import rgb2gray
from Kmeans import get_colors
from utils_data import visualize_retrieval
import numpy as np

def visualize_averages(knn, test_imgs, uniqueShapes):
    shape_pred = knn.predict(rgb2gray(test_imgs), 10)
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
            shapeAv = shapeAv[20:-40, 10:-10]
        elif shape == "Heels":
            shapeAv = shapeAv[17:-20,7:-7]
        shapeAverage.append(shapeAv)
    
    visualize_retrieval(shapeAverage, len(shapeAverage))


def findBestDEC(km, minDEC, maxDEC, pointCount,test_color_labels, get_color_accuracy, repetitions=3):
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
