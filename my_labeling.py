__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from Kmeans import KMeans, Options
from utils_data import read_dataset, read_extended_dataset, crop_images
import utils_data


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    
    defaults: Options = {
        "km_init": "random",
        "verbose": False,
        "tolerance": 0,
        "opt_DEC": 0.8,
        "max_iter": 100,
        "fitting": "WCD",
    }
    first = imgs[0]
    km = KMeans(first, 1, defaults)
    km.find_bestK(10)
    print("Best K = ", km.K)
    km.fit()
    utils_data.visualize_k_means(km, first.shape)

