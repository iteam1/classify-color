'''
python3 utils/post_process.py assets/red.jpg
'''
import sys
import cv2
import random
import numpy as np

IMG_PATH = sys.argv[1]
DIM = 224

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

    # shuffle pixels
    random.shuffle(pixels)
    
    print('total pixels:',H*W,'nonzero pixels:',len(pixels), round(len(pixels)/(H*W),2),'%')
    
    box = np.array(pixels[:DIM*DIM])
    
    box = box.reshape(DIM,DIM,C)
    
    cv2.imwrite('box.png',cv2.cvtColor(box,cv2.COLOR_RGB2BGR))
    
        