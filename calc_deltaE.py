'''
python3 classify_color.py data/COLOR/WHITE/4.jpg
'''
import os
import sys
import cv2
import colour
import numpy as np
from scipy.stats import itemfreq

IMG_PATH = sys.argv[1]
IMG_SIZE = 640
n_colors=3 #number of clusters
#number of iterations
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
#initialising centroid
flags = cv2.KMEANS_RANDOM_CENTERS

COLORS  = {'WHITE':(255,255,255),
           'BLACK':(0,0,0),
           'RED':(255,0,0),
           'GREEN':(0,255,0),
           'YELLOW':(255,255,0),
           'BLUENAVY':(0,0,128),
           'BLUE':(0,0,255),
           'PURPLE':(127,70,171),
           }

if __name__ == "__main__":
    
    img = cv2.imread(IMG_PATH)
    
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    img_arr=np.float32(img_rgb)
    
    #reshaping the image to a linear form with 3-channels
    pixels=img_arr.reshape((-1,3))
    
    #number of iterations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    
    #initialising centroid
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    #applying k-means to detect prominant color in the image
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    
    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    
    #detecting the centroid with densest cluster  
    dominant_rgb = palette[np.argmax(itemfreq(labels)[:, -1])]
    dominant_lab = cv2.cvtColor( np.uint8([[dominant_rgb]]) , cv2.COLOR_RGB2LAB)
    
    keys = list(COLORS.keys())
    
    for key in keys:
        current_color = COLORS[key]
        current_lab = cv2.cvtColor(np.uint8([[current_color]]) , cv2.COLOR_RGB2LAB)
        deltaE = delta_E = colour.delta_E(dominant_lab, current_lab)
        print(key,deltaE)

    
    
    
