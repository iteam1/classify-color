'''
python3 utils/hsv_tool.py data/plallete.png
'''

import numpy as np
import cv2
import sys

def empty(i):
    pass

def on_trackbar(val):
    global img,hsv,res
    # get H,S,V value from trackbar
    hue_min = cv2.getTrackbarPos("Hue Min", "TrackedBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackedBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackedBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackedBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackedBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackedBars")
    scale   = cv2.getTrackbarPos("Scale", "TrackedBars")
    # get current mask of HSV range
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img,img,mask=mask)
    # reshape the image
    h,w,c = res.shape
    h = int(h*scale/100)
    w = int(w*scale/100)
    resized = cv2.resize(res,(0,0),fx=scale/100,fy=scale/100)
    # display result
    cv2.imshow("HSV",resized)

# read input the images
img_paths = sys.argv[1:] # the argument after program's name
print(f"Reading {len(img_paths)} images")
imgs = [] # list of input image
for img_path in img_paths:
    img = cv2.imread(img_path)
    imgs.append(img)

# stack the images
img = cv2.hconcat(imgs)

# convert to hsv space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# create window
cv2.namedWindow("TrackedBars")
cv2.resizeWindow("TrackedBars", 640, 240)
# create trackbars
cv2.createTrackbar("Hue Min", "TrackedBars", 0, 179, on_trackbar)
cv2.createTrackbar("Hue Max", "TrackedBars", 179, 179, on_trackbar)
cv2.createTrackbar("Sat Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Sat Max", "TrackedBars", 255, 255, on_trackbar)
cv2.createTrackbar("Val Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Val Max", "TrackedBars", 255, 255, on_trackbar)
cv2.createTrackbar("Scale", "TrackedBars",10, 100, on_trackbar)

# show some stuff
on_trackbar(0)

# wait until user press any key
k = cv2.waitKey()

#destroy All Windows
cv2.destroyAllWindows()
