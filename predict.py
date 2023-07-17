'''
python3 predict.py DATASET/UNSEEN/BLUE/color_classify_13072023131752_media_6_Blue.jpg
'''
import os
import sys
import cv2
import json
import pickle
import numpy as np

NUM_CLUSTER = 5 # number of cluster
DIM = 240
IMG_PATH = sys.argv[1]

def preprocess(im):
    
    pixels = []
    
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    H,W,C = im.shape
    mask = np.all(im != [0,0,0],axis=2)
    
    # collect nonzero pixels
    for i in range(H):
        for j in range(W):
            if mask[i,j]:
                pixels.append(im[i,j])
    
    print('total pixels:',H*W,'nonzero pixels:',len(pixels))
    
    # convert to np.float32
    Z = np.float32(np.array(pixels))
    # define criteria, number of clusters and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # kmeans
    ret, labels, centroids = cv2.kmeans(Z, NUM_CLUSTER, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centroids = np.uint8(centroids)
    
    dict = []
    l = []
    vals = list(set(list(labels.flatten())))
    for val in vals:
        qua = list(labels.flatten()).count(val)
        des = round(qua/(H*W),2)
        qua_DIM = int(DIM*DIM*des)
        l.append(qua_DIM)
        dict.append({'color':centroids[val],'density':des,'quantity':qua,f'quantity_{DIM}':qua_DIM})
        # print(' color:',centroids[val],' density:',des,' quantity:',qua,f' quantity_{DIM}:',qua_DIM)
    
    # convert to box
    box = []
    for m in range(NUM_CLUSTER - 1):
        current_color = dict[m]['color']
        q = dict[m][f'quantity_{DIM}']
        current_box = [current_color]*q
        box += current_box
    # append the last cluster
    current_color = dict[-1]['color']
    current_box = [current_color]*(DIM*DIM - len(box))
    box += current_box
    
    box = np.array(box).reshape((DIM,DIM,3))
    
    return box

if __name__ =="__main__":
    
    # load saved model
    loaded_model = pickle.load(open('model_color.sav','rb'))
    print('Model loaded.')

    # load labels
    with open('labels.json','r') as f:
        labels = json.load(f)
    
    # read image
    img = cv2.imread(IMG_PATH)  
    
    box = preprocess(img)
    
    # predict
    x = box/255.0
    x = x.flatten()
    x = x.reshape(1,-1)
    
    pred = loaded_model.predict(x)
    print(labels[str(pred[0])])