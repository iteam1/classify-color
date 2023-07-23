'''
python3 visualize_data.py data/COLOR6
'''
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

DATA_PATH = sys.argv[1] # image path
NUM_CLUSTER = 3 # number of cluster
MAX_VALUE = 255

colors = os.listdir(DATA_PATH)

dataset = {}

RGBs = []

# collect data
for color in colors:

    print('Parsing ',color)

    # list all image
    imgs = os.listdir(os.path.join(DATA_PATH,color.upper()))

    # init
    R = []
    G = []
    B = []

    for n,img_name in enumerate(imgs):

        # init
        pixels = []

        # read the image
        img = cv2.imread(os.path.join(DATA_PATH,color.upper(),img_name))
        h,w,c = img.shape
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

        # remove zeropixel
        mask = np.all(img != [0,0,0],axis = 2)
        for i in range(h):
            for j in range(w):
                if mask[i,j]:
                    pixels.append(img[i,j])

        total_pixels = h*w
        zero_pixels = h*w-len(pixels)

        print(n,img_name,'total pixels:',total_pixels,' zero_pixels:',zero_pixels,\
        round((total_pixels-zero_pixels)/total_pixels,2))

        # convert to numpy array
        Z = np.array(pixels)

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # kmeans
        ret, labels, centers = cv2.kmeans(Z, NUM_CLUSTER, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.uint8)

        # collect R,G,B value
        for k in range(centers.shape[0]):
            current_color = centers[k]
            R.append(current_color[0])
            G.append(current_color[1])
            B.append(current_color[2])

    RGBs.append([R,G,B])

# visualize data
fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')

for i in range(len(colors)):
    R,G,B = RGBs[i]

    # creating plot
    ax.scatter3D(R,G,B,color=colors[i])
    plt.title('HSV 3D scatter plot')
    ax.set_xlim(0,MAX_VALUE)
    ax.set_ylim(0,MAX_VALUE)
    ax.set_xlabel('R axis')
    ax.set_ylabel('G axis')
    ax.set_zlabel('B axis')

# show plot
plt.show()
