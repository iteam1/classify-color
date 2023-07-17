'''
python3 test.py DATASET/UNSEEN/PURPLE
'''
import os
import sys
import cv2
import json
import pickle
import numpy as np

NUM_CLUSTER = 5 # number of cluster
DIM = 256
THRESH = 0.65
IMGS_PATH = sys.argv[1]

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
        
    imgs = os.listdir(IMGS_PATH)
    
    for file in imgs:
        
        pred = 'UNKNOWN'
    
        # read image
        img = cv2.imread(os.path.join(IMGS_PATH,file))  
        
        box = preprocess(img)
        
        # predict
        x = box#/255.0
        x = x.flatten()
        x = x.reshape(1,-1)
        
        pred_propa = loaded_model.predict_proba(x)
        
        # Get the probability for the first instance in the test set
        instance_prob = pred_propa[0]
        
        idx =np.argmax(instance_prob)
        
        # Get the maximum probability
        max_prob = max(instance_prob)

        # Calculate the confidence score
        confidence_score = max_prob / sum(instance_prob)
        
        if confidence_score > THRESH:
            pred = labels[str(idx)]

        print(os.path.join(IMGS_PATH,file),pred,labels[str(idx)],'confidence: ',confidence_score)