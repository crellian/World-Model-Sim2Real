import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import numpy as np
import cv2
import random
import bisect
from tensorflow.keras.utils import Sequence
root_dir = "/home2/random_bev_carla/rgb_bev/Town01_0"

for root, subdirs, files in os.walk(root_dir):
    for f in files:
        if 'observation_rgb.npy' in f:
            print(root)
            rgb = np.load(os.path.join(root, f), mmap_mode='r')
            print(rgb.shape)
        elif 'observation.npy' in f:
            bev = np.load(os.path.join(root, f), mmap_mode='r')[:,:,:,0]
        elif 'label' in f:
            l = np.load(os.path.join(root, f))

anchors = []
for i in range(10):
    img = cv2.imread("manual_label/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
    anchors.append(img)

model = models.load_model("/lab/kiran/img2cmd_data/model/dr")
#reconstructed_model.compile(loss="sparse_categorical_crossentropy", metrics=['accuracy'])

cnt = 0
for i in range(len(bev)):
    #ground truth
    #cv2.imshow("bev", bev[i])
    #cv2.imshow("anchor", anchors[l[i]])

    #predict
    im = np.expand_dims(rgb[i], axis=0) / 255.0
    probs = model.predict(im, verbose=0)
    id = np.argmax(probs)
    #cv2.imshow("pred", anchors[id])
    #cv2.waitKey(100)
    if id == l[i]:
        cnt+=1
print(cnt/len(bev))
