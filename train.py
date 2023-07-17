'''
python3 train.py dst
'''
import os
import sys
import cv2
import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

le = LabelEncoder()

DATA_PATH = sys.argv[1]

colors = os.listdir(DATA_PATH)

X = []
y = []

# collect data
for i,color in enumerate(colors):
    path = os.path.join(DATA_PATH,color)
    for file in os.listdir(path):
        im = cv2.imread(os.path.join(path,file))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = im/255.0
        im = im.flatten()
        X.append(im)
        y.append(i)

X = np.array(X)
y = np.array(y)

print("Summary")
print("X: ",X.shape)
print("y: ",y.shape)
# shuffle data
X,y = shuffle(X,y)

# split train val subset
print('Training')
y = le.fit_transform(y)
# xgb.XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
model = xgb.XGBClassifier(max_depth=5)  #DecisionTreeClassifier(max_depth=5)

model.fit(X, y)
preds = model.predict(X)
  
# accuracy on X_test
# print("Validation report:",classification_report(y,preds))
print('Accuracy:',accuracy_score(y, preds))

# creating a confusion matrix
cm = confusion_matrix(y, preds)
print(cm)
    




