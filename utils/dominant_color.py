'''
python3 utils/dominant_color.py assets/black.jpg
'''
import os
import sys
import cv2
import numpy as np

IMG_PATH = sys.argv[1] # image path
NUM_CLUSTER = 3 # number of cluster

def create_colorbar(label,center):
    '''
    create colorbar
    '''
    bars = []
    SIZE = 100

    pixel_labels = list(label.flatten())
    num_cluster = len(set(pixel_labels))
    # print('num_cluster:',num_cluster)

    value_dict = {}
    pixel_counts = []
    total = 0
    color = (127,50,127)
    font_size = 0.3
    thickness = 1

    for i in range(num_cluster):
        pixel_no = pixel_labels.count(i)
        pixel_counts.append(pixel_no)
        #print('label:',i,'value:',center[i],'quantity',pixel_no)
        value_dict[pixel_no] = i
        total += pixel_no

    # sort values
    sorted_pixel_counts = sorted(pixel_counts)
    #print(sorted_pixel_counts

    # create colorbar
    for count_value in sorted_pixel_counts:
        bar = np.zeros((SIZE,SIZE,3),np.uint8)
        # get current image
        current_index = value_dict[count_value]
        current_color = np.uint8(center[current_index])
        #print('index:',current_index,'color:',current_color)
        bar[:] = current_color
        cv2.putText(bar,str(round(count_value/total,2)),(10,30), cv2.FONT_HERSHEY_SIMPLEX,\
        font_size, color, 1, cv2.LINE_AA)
        cv2.putText(bar,str(current_color),(10,50), cv2.FONT_HERSHEY_SIMPLEX,\
        font_size, color, thickness, cv2.LINE_AA)
        bars.append(bar)

    colorbar = np.hstack(bars)

    return colorbar

# read the image
img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# reshape the image
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters and apply kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# kmeans
ret, label, center = cv2.kmeans(Z, NUM_CLUSTER, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# parsing color
colorbar = create_colorbar(label,center)

# now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res = res.reshape(img.shape)
res = np.hstack([img,res])

# display
cv2.imshow('res',cv2.cvtColor(res,cv2.COLOR_RGB2BGR))
cv2.imshow('colorbar',cv2.cvtColor(colorbar,cv2.COLOR_RGB2BGR))
k = cv2.waitKey()
cv2.destroyAllWindows()
