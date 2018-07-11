# this is a cleaned-up and simplified version of face-recognition.py

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

warnings.filterwarnings('ignore')
image_folder = 'images'

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

metadata = [] # image metadata: dir, name, file_name
for i in os.listdir(image_folder):
    for f in os.listdir(os.path.join(image_folder, i)):
        metadata.append(IdentityMetadata(image_folder, i, f))
metadata = np.array(metadata)

alignment = AlignDlib('models/landmarks.dat')

# Embedding vectors can now be calculated by feeding the aligned and scaled images into the pre-trained network.
# embedded = np.zeros((metadata.shape[0], 128))
# Ilya: generate 128 point embeddings for each image. This part takes a long time.
# for i, m in enumerate(metadata):
#     img = load_image(m.image_path())
#     img = align_image(img, alignment)
#     # scale RGB values to interval [0,1]
#     img = (img / 255.).astype(np.float32)
#     # obtain 128 point embedding vector for each image
#     embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
# pickle.dump(embedded, open('embedded.p','w'))

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
# example_image = load_image( metadata[test_idx][example_idx].image_path() )
example_prediction = svc.predict( [embedded[test_idx][example_idx]] )
example_identity = encoder.inverse_transform(example_prediction)[0]

print('Recognized as ' + example_identity)



