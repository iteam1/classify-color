'''
python3 utils/post_process.py assets/red.jpg
'''
import sys
import cv2
import random
import numpy as np

IMG_PATH = sys.argv[1]
DIM = 224
NUM_CLUSTER = 3

def myfunction():
  return 0.1

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
                
    # clustering pixels

    Z = np.float32(np.array(pixels))     # convert to np.float32
    # define criteria, number of clusters and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # kmeans
    ret, labels, centroids = cv2.kmeans(Z, NUM_CLUSTER, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centroids = np.uint8(centroids)
    
    print(centroids)

    # shuffle pixels
    random.shuffle(pixels)
    
    print('total pixels:',H*W,'nonzero pixels:',len(pixels), round(len(pixels)/(H*W),2),'%')
    
    box = np.array(pixels[:DIM*DIM])
    
    box = box.reshape(DIM,DIM,C)
    
    cv2.imwrite('box.png',cv2.cvtColor(box,cv2.COLOR_RGB2BGR))
    
        