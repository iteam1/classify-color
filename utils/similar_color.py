'''
python3 utils/similar_color.py data/COLOR2/BLACK/BLACK49_0.jpg
'''
import sys
import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2lab, deltaE_cie76

IMG_PATH = 'https://i.stack.imgur.com/npnrv.png' #sys.argv[1]
rgb = io.imread(IMG_PATH)
lab = rgb2lab(rgb)

green = [0, 160, 0]
magenta = [120, 0, 140]

threshold_green = 15
threshold_magenta = 20

green_3d = np.uint8(np.asarray([[green]]))
magenta_3d = np.uint8(np.asarray([[magenta]]))

dE_green = deltaE_cie76(rgb2lab(green_3d), lab)
dE_magenta = deltaE_cie76(rgb2lab(magenta_3d), lab)

rgb[dE_green < threshold_green] = green_3d
rgb[dE_magenta < threshold_magenta] = magenta_3d

cv2.imshow('rgb',rgb)

cv2.waitKey()

cv2.destroyAllWindows()
