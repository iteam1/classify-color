import os
import sys
import cv2
import json
import pickle
import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

DATA_PATH = sys.argv[1]

colors = os.listdir(DATA_PATH)

X = []
y = []

labels = {}
# collect data
for i,color in enumerate(colors):
    labels[i] = color
    path = os.path.join(DATA_PATH,color)
    for file in os.listdir(path):
        im = cv2.imread(os.path.join(path,file))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        #im = im/255.0
        im = im.flatten()
        X.append(im)
        y.append(i)

# export labels
with open('soln1/labels.json','w') as f:
    json.dump(labels,f)

X = np.array(X)
y = np.array(y)

# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
y = label_encoder.transform(y)

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=42)

print("Summary")
print("X_train: ",X_train.shape)
print("y_train: ",y_train.shape)
print("X_val: ",X_val.shape)
print("y_val: ",y_val.shape)

# shuffle data
X_train,y_train = shuffle(X_train,y_train)
X_val,y_val = shuffle(X_val,y_val)

# split train val subset
print('Training')
# xgb.XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
model = xgb.XGBClassifier(max_depth=10)  # DecisionTreeClassifier(max_depth=5)

# train model
model.fit(X_train, y_train)

# accuracy on X_test
# print("Validation report:",classification_report(y,preds))
preds = model.predict(X)
print('Accuracy:',accuracy_score(y, preds))

# creating a confusion matrix
cm = confusion_matrix(y, preds)
print(cm)

 # save the model to disk
filename = f'soln1/model_classify_color.sav'
pickle.dump(model, open(filename, 'wb'))
