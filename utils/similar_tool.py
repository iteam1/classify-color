'''
Lab parser
Support function horizontal stack same size image
python3 utils/similar_tool.py assets/npnrv.png
'''
import cv2
import sys
import numpy as np
from skimage import io
from skimage.color import rgb2lab, deltaE_cie76

THRESH = 10

def empty(i):
    pass

def on_trackbar(val):
    global lab,rgb

    # get L,a,b value from trackbar
    L_value = cv2.getTrackbarPos("Light", "TrackedBars")
    a_value = cv2.getTrackbarPos("a", "TrackedBars")
    b_value = cv2.getTrackbarPos("b", "TrackedBars")
    scale   = cv2.getTrackbarPos("Scale", "TrackedBars")

    # get current mask of Lab range
    lab_value = np.uint8(np.asarray([[L_value,a_value,b_value]]))

    # calculate deltaE
    dE = deltaE_cie76(rgb2lab(lab_value), lab)
    
    res = rgb.copy()
    res[dE < THRESH] = lab_value
    
    # reshape the image
    h,w,c = lab.shape
    h = int(h*scale/100)
    w = int(w*scale/100)
    resized = cv2.resize(res,(0,0),fx=scale/100,fy=scale/100)

    # display result
    cv2.imshow("LAB",resized)

# read input the images
IMG_PATH = sys.argv[1]
rgb = io.imread(IMG_PATH)
lab = rgb2lab(rgb)

# create window
cv2.namedWindow("TrackedBars")
cv2.resizeWindow("TrackedBars", 640, 240)
# create trackbars
cv2.createTrackbar("Light", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("a", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("b", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Scale", "TrackedBars",10, 100, on_trackbar)

# show some stuff
on_trackbar(0)

# wait until user press any key
k = cv2.waitKey()

cv2.destroyAllWindows()