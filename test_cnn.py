import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import numpy as np
import cv2

root_dir = "/home/carla/img2cmd/anchorimgs"

imgs = []
for root, subdirs, files in os.walk(root_dir):
    for f in files:
        img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)
        imgs.append(img)

reconstructed_model = models.load_model("cnn")

print(reconstructed_model.predict(np.array(imgs)))
