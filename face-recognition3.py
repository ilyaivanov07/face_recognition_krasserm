# this is a follow up on face-recognition2.py.
# attempt to use a video file based on face-recognition2.py

import os
from model import create_model
import numpy as np
import os.path
import matplotlib.pyplot as plt
from align import AlignDlib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import warnings
import time
from utils import IdentityMetadata, load_image, align_image
import pickle
import cv2

image_folder = 'images'
metadata = pickle.load(open('metadata.p'))
embedded = pickle.load(open('embedded.p'))
targets = np.array([m.name for m in metadata])
encoder = LabelEncoder()
encoder.fit(targets)

# Numerical encoding of identities
y = encoder.transform(targets)

train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

# 50 train examples of 10 identities (5 examples each)
x_train = embedded[train_idx]
y_train = y[train_idx]

svc = LinearSVC()
svc.fit(x_train, y_train)

example_idx = 0
example_prediction = svc.predict( [embedded[test_idx][example_idx]] )
print (example_prediction)
example_identity = encoder.inverse_transform(example_prediction)[0]

print('Recognized as ' + example_identity)



