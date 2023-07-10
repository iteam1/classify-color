'''
python3 batch_test.py data
'''
import os
import sys
import cv2
import numpy as np

IMGS_PATH = sys.argv[1] # image path
NUM_CLUSTER = 5 # number of cluster

COLOR_CLASSES = {(255,0,0):'Red',
(0,0,0):'Black',
(255,255,255):'White',
(0,255,0):'Green',}

# custom threshold
THRESH_CLASSES ={'Red':0.4,
                 'Black':0.6,
                 'White':0.6,
                 'Green':0.4}

def classify_dominant(img):
    '''
    function classify color
    '''
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
                        'label':COLOR_CLASSES[closest_color]}

        # update current color
        color_dict[i] = current_color

    return color_dict

def post_proces(res):
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

    return pred

if __name__ == "__main__":
    
    imgs = os.listdir(IMGS_PATH)
    
    for i,img_name in enumerate(imgs):
        # read the image
        img = cv2.imread(os.path.join(IMGS_PATH,img_name))

        # parsing
        res = classify_dominant(img)

        # post process
        pred = post_proces(res)

        print(i,img_name,'Prediction:',pred)