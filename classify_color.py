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

# read the image
img = cv2.imread(IMG_PATH)
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

classification = []
for color in center:
    # find the closest color
    closest_color = min(COLOR_CLASSES, key=lambda x: np.linalg.norm(color -x))
    classification.append(closest_color)

print('Dominant colors is the image')
for color in classification:
    print('color:', COLOR_CLASSES[color])
