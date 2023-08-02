'''
Lab parser
Support function horizontal stack same size image
python3 utils/lab_tool.py assets/npnrv.png
'''
import cv2
import sys
import numpy as np

def empty(i):
    pass

def on_trackbar(val):
    global img,res

    # get L,a,b value from trackbar
    L_min = cv2.getTrackbarPos("Light Min", "TrackedBars")
    L_max = cv2.getTrackbarPos("Light Max", "TrackedBars")
    a_min = cv2.getTrackbarPos("a channel Min", "TrackedBars")
    a_max = cv2.getTrackbarPos("a channel Max", "TrackedBars")
    b_min = cv2.getTrackbarPos("b channel Min", "TrackedBars")
    b_max = cv2.getTrackbarPos("b channel Max", "TrackedBars")
    scale   = cv2.getTrackbarPos("Scale", "TrackedBars")

    # get current mask of Lab range
    lower = np.array([L_min, a_min, b_min])
    upper = np.array([L_max, a_max, b_max])
    mask = cv2.inRange(res, lower, upper)
    roi = cv2.bitwise_and(img,img,mask=mask)

    # reshape the image
    h,w,c = roi.shape
    h = int(h*scale/100)
    w = int(w*scale/100)
    resized = cv2.resize(roi,(0,0),fx=scale/100,fy=scale/100)

    # display result
    cv2.imshow("LAB",resized)

# read input the images
img_path = sys.argv[1]
img = cv2.imread(img_path)

# convert to lab space
res = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# create window
cv2.namedWindow("TrackedBars")
cv2.resizeWindow("TrackedBars", 640, 240)
# create trackbars
cv2.createTrackbar("Light Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("Light Max", "TrackedBars", 255, 255, on_trackbar)
cv2.createTrackbar("a channel Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("a channel Max", "TrackedBars", 255, 255, on_trackbar)
cv2.createTrackbar("b channel Min", "TrackedBars", 0, 255, on_trackbar)
cv2.createTrackbar("b channel Max", "TrackedBars", 255, 255, on_trackbar)
cv2.createTrackbar("Scale", "TrackedBars",10, 100, on_trackbar)

# show some stuff
on_trackbar(0)

# wait until user press any key
k = cv2.waitKey()

cv2.destroyAllWindows()
