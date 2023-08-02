'''
python3 utils/post_process.py assets/red.jpg
'''
import sys
import cv2
import random
import numpy as np
from skimage import io
from skimage.color import rgb2lab, deltaE_cie76

IMG_PATH = sys.argv[1]
DIM = 224 # dimension of box
NUM_CLUSTER = 3 # number of cluster
THRESHOLD = 20 # threshold for similar color

if __name__ == "__main__":
    
    # read the image
    img = cv2.imread(IMG_PATH)
    
    # convert to RGB
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    H,W,C = img_rgb.shape
    
    # count nonzero pixels
    pixels = []
    
    mask = np.all(img_rgb != [0,0,0],axis=2)

    # collect nonzero pixels
    for i in range(H):
        for j in range(W):
            if mask[i,j]:
                pixels.append(img_rgb[i,j])
                
    print('total pixels:',H*W,'nonzero pixels:',len(pixels), round(len(pixels)/(H*W),2),'%')

    # convert to lab
    lab_pixels = rgb2lab(pixels)
                
    # clustering pixels
    Z = np.float32(np.array(pixels))     # convert to np.float32
    
    # define criteria, number of clusters and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # kmeans
    ret, labels, centroids = cv2.kmeans(Z, NUM_CLUSTER, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centroids = np.uint8(centroids)
    
    labels = list(labels.flatten())
    label_keys = list(set(labels))
    label_dict = {key:[] for key in label_keys}
    label_counts = [labels.count(key) for key in label_keys]
    label_sizes = [int(DIM*DIM*label_counts[i]/len(labels)) for i in range(len(label_counts)-1)]
    label_sizes.append(DIM*DIM - sum(label_sizes)) # the last cluster

    for i,pixel in enumerate(pixels):
        # update label_dict
        key = labels[i]
        label_dict[key].append(pixels[i])

    collected_pixels = []
    for i in range(len(label_keys)):
        tmp  = label_dict[i][:label_sizes[i]]
        collected_pixels += tmp
        
    # shuffle pixels
    random.shuffle(collected_pixels)
    
    box = np.array(collected_pixels)
    
    box = box.reshape(DIM,DIM,C)
    
    cv2.imwrite('box.png',cv2.cvtColor(box,cv2.COLOR_RGB2BGR))
    
        