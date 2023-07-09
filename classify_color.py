'''
python3 classify_color.py data/black.jpg
'''
import os
import sys
import cv2
import numpy as np

IMG_PATH = sys.argv[1] # image path
NUM_CLUSTER = 3 # number of cluster
COLOR_CLASSES = {(255,0,0):'Red',
(0,0,0):'Black',
(255,255,255):'White'}

def classify_dominant(img):

    color_dict = {}
    pixel_counts = []
    total_pixel = 0

    # convert image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # reshape the image
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # kmeans
    ret, label, center = cv2.kmeans(Z, NUM_CLUSTER, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    pixel_labels = list(label.flatten())
    total_pixel = len(pixel_labels)
    num_cluster = len(set(pixel_labels))
    print('num_cluster:',num_cluster,'total pixel:',total_pixel)

    for i in range(num_cluster):
        # get current color
        color = center[i]

        # find the closest color
        closest_color = min(COLOR_CLASSES, key=lambda x: np.linalg.norm(color -x))

        current_color = {'value':np.uint8(color),\
                        'percent':round(pixel_labels.count(i)/total_pixel,2),\
                        'class':COLOR_CLASSES[closest_color]}

        # update current color
        color_dict[i] = current_color

    return color_dict

if __name__ == "__main__":
    # read the image
    img = cv2.imread(IMG_PATH)

    # parsing
    res = classify_dominant(img)

    print(res)
