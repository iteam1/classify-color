'''
python3 batch_test.py data/color/4
'''
import os
import sys
import json
import cv2
import numpy as np

IMGS_PATH = sys.argv[1] # image path

NUM_CLUSTER = 3 # number of cluster

MAX_DISTANCE = 50 # 255^2*3

COLOR_CLASSES = {(255,0,0):'Red',
                 (100,10,10):'RedRuby',
                 (250,100,100):'RedCoral',
                 (0,0,0):'Black',
                 (66,78,92):'BlackGray',
                 (230,230,220):'White',
                 (220,220,210):'WhitePearl',
                 (230,230,170):'Yellow',
                 (208,207,140):'YellowKhali',
                 (180,180,160):'GoldMetalic',
                 (10,50,120):'Blue',
                 (100,200,250):'OceanBlue',
                 (140,220,220):'ColumbiaBlue',
                 (30,80,120):'DenimBlue',
                 (165,214,195):'GreenMint',
                 (200,230,160):'GreenLawn',
                 (212,231,212):'GreenTea',
                 }

def classify_dominant(img):
    '''
    function classify color
    '''
    pixels = []
    # convert image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    H,W,c = img.shape
    
    # get mask
    res = np.all(img != [0,0,0],axis=2)
    zero_pixels = H*W - np.count_nonzero(res)
    msg = 'total_pixels: '+str(H*W)+' zero_pixels: '+str(zero_pixels)
    # collect nonzero pixels
    for y in range(H):
        for x in range(W):
            if res[y,x]:
                pixels.append(img[y,x])
    
    rec = np.array(pixels)
    
    # convert to np.float32
    Z = np.float32(rec)
    
    # define criteria, number of clusters and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # kmeans
    ret, label, center = cv2.kmeans(Z, NUM_CLUSTER, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    pixel_labels = list(label.flatten())
    total_pixel = len(pixel_labels)
    
    for i in range(NUM_CLUSTER):
        # get current color
        color = center[i]
        
        # find the closest color
        closest_color = min(COLOR_CLASSES, key=lambda x: np.linalg.norm(color -x))
        
        # calculate distance
        distance = np.linalg.norm(color - closest_color)
        coff = round(1 - distance/MAX_DISTANCE,2)
        S = round(pixel_labels.count(i)/total_pixel,2)
        
        msg = str(i) + ' value: ' + str(np.uint8(color)) + ' pred: ' + COLOR_CLASSES[closest_color] \
            + ' distance: '+ str(round(distance,2))+ ' coffident: ' + str(coff) + ' percent: ' + str(S) \
                + ' dS: ' + str(round(S*distance,2))
        
        print(msg,)

if __name__ == "__main__":
    
    imgs = os.listdir(IMGS_PATH)
    
    for i,img_name in enumerate(imgs):
        
        print(os.path.join(IMGS_PATH,img_name))
        
        # read the image
        img = cv2.imread(os.path.join(IMGS_PATH,img_name))

        # parsing
        classify_dominant(img)
        
        print()