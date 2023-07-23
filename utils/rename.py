'''
python3 utils/rename.py data/COLOR6
'''
import time
import os
import sys

DATA_PATH = sys.argv[1]

labels = os.listdir(DATA_PATH)

for label in labels:
    print('label:',label)
    prefix = str(time.time())[-2:]
    path = os.path.join(DATA_PATH,label)
    imgs = os.listdir(path)
    for i,img in enumerate(imgs):
        os.rename(os.path.join(path,img),os.path.join(path,f'{label}{prefix}_{i}.jpg'))
