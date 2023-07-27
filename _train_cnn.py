import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models
import os
import numpy as np
from tensorflow.keras.utils import Sequence

root_dir = "/home/tmp/kiran/random_bev_carla/rgb_bev"

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(84, 84, 3)))
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

train_images = None
train_labels = None
for root, subdirs, files in os.walk(root_dir):
    for f in files:
        if 'observation_rgb.npy' in f:
            print(root)
            rgb = np.load(os.path.join(root, f), mmap_mode='r')
            print(rgb.shape)
            train_images = rgb if train_images is None else np.concatenate((train_images, rgb))
        
        elif 'label' in f:
            l = np.load(os.path.join(root, f))
            train_labels = l if train_labels is None else np.concatenate((train_labels, le))
    if 'label.npy' in files:
        break

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size] / 255.0
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def on_epoch_end(self):
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.y = self.y[p]

train_gen = DataGenerator(train_images, train_labels, 256)

history = model.fit(train_gen, epochs=10)
#history = model.fit(train_images / 255.0, train_labels, epochs=10, batch_size=256)

model.save('model/cnn', save_format='tf')
