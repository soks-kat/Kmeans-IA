from utils import rgb2gray
from Kmeans import get_colors
from utils_data import visualize_retrieval
import numpy as np
from multiprocessing import Pool

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


