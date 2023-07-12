'''
python3 utils/batch_test.py data/color/gold
'''
import os
import sys
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
                 (154,227,104):'GreenLight',
                 }

THRESH_CLASSES ={'Red':0.5,
                 'RedRuby':0.5,
                 'RedCoral':0.5,
                 'Black':0.5,
                 'BlackGray':0.5,
                 'White':0.5,
                 'WhitePearl':0.5,
                 'Yellow':0.5,
                 'YellowKhali':0.5,
                 'GoldMetalic':0.5,
                 'Blue':0.5,
                 'OceanBlue':0.35,
                 'ColumbiaBlue':0.35,
                 'DenimBlue':0.35,
                 'GreenMint':0.4,
                 'GreenLawn':0.4,
                 'GreenTea':0.4,
                 'GreenLight':0.4,
                 }

LABELS = ['Red','Black','White','Blue','Yellow','Gold','Green']

SUMMARY = {}
labels = list(THRESH_CLASSES.keys())
for label in LABELS:
    SUMMARY[label] = 0
SUMMARY['Unknown'] = 0

def classify_dominant(img):
    '''
    function classify color
    '''
    color_dict = {}
    pixels = []
    
    # convert image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    H,W,c = img.shape
    
    # get mask
    res = np.all(img != [0,0,0],axis=2)
    zero_pixels = H*W - np.count_nonzero(res)
    msg = 'total_pixels: '+str(H*W)+' zero_pixels: '+str(zero_pixels)
    print(msg)
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
    total_nonzero_pixels = len(pixel_labels)
    num_cluster = len(set(pixel_labels))
    msg = 'num_cluster:'+ str(num_cluster) + ' total nonzero pixels:' + str(total_nonzero_pixels)
    print(msg)

    for i in range(num_cluster):
        # get current color
        color = center[i]

        # find the closest color
        closest_color = min(COLOR_CLASSES, key=lambda x: np.linalg.norm(color -x))
        
        # calculate distance
        distance = np.linalg.norm(color - closest_color)
        coff = round(1 - distance/MAX_DISTANCE,2)

        current_color = {'value':np.uint8(color),
                         'percent':round(pixel_labels.count(i)/(total_nonzero_pixels),2),\
                             'distance':distance,\
                                 'coffident':coff,\
                                     'label':COLOR_CLASSES[closest_color]}

        # update current color
        color_dict[i] = current_color

    return color_dict

def post_process(res):
    '''
    post process result
    '''
    pred = 'Unknown'
    out = {}

    inv_map = {v: k for k, v in COLOR_CLASSES.items()}
    labels = list(inv_map)
    # print(labels)

    for label in labels:
        out[label] = 0.0

    keys = list(res.keys())

    for key in keys:
        current_color = res[key]
        label = current_color['label']
        value = current_color['percent']
        out[label] += value
        
    print(out)
    
    # predict
    for label in labels:
        current_thresh = THRESH_CLASSES[label]
        percent_value = out[label]
        if percent_value >= current_thresh:
            pred = label
            break
        
    # preprocess
    for label in LABELS:
        if label in pred:
            pred = label
            break

    return pred

if __name__ == "__main__":
    
    imgs = os.listdir(IMGS_PATH)
    
    for i,img_name in enumerate(imgs):
        # read the image
        img = cv2.imread(os.path.join(IMGS_PATH,img_name))

        # parsing
        res = classify_dominant(img)

        # post process
        pred = post_process(res)
        
        SUMMARY[pred] += 1

        print(i,os.path.join(IMGS_PATH,img_name),'Prediction:',pred,'\n')
        
    print('SUMMARY:',SUMMARY)