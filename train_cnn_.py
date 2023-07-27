import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models
import os
import bisect
import random
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

train_images = []
train_labels = []
each_len = []
for root, subdirs, files in os.walk(root_dir):
    for f in files:
        if 'observation_rgb.npy' in f:
            print(root)
            rgb = np.load(os.path.join(root, f), mmap_mode='r')
            print(rgb.shape)
            train_images.append(rgb) 
        
        elif 'label' in f:
            l = np.load(os.path.join(root, f))
            train_labels.append(l)
    if 'label.npy' in files:
        break

for i in range(len(train_labels)):                    
    if len(each_len) == 0:
        each_len.append(train_labels[i].shape[0])
    else:
        each_len.append(train_labels[i].shape[0] + each_len[-1])


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, each_len, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.each_len = each_len

    def __len__(self):
        return int(np.ceil(self.each_len[-1] / float(self.batch_size)))

    def __getitem__(self, idx):
        idx = idx * self.batch_size
        file_ind = bisect.bisect_right(self.each_len, idx)
        if file_ind == 0:
            im_ind = idx
        else:
            im_ind = idx - self.each_len[file_ind-1]
        #print(idx, file_ind, im_ind, self.each_len)

        batch_x = self.x[file_ind][im_ind:im_ind + self.batch_size] / 255.0
        batch_y = self.y[file_ind][im_ind:im_ind + self.batch_size]
        return batch_x, batch_y

    def on_epoch_end(self):
        l = list(zip(self.x, self.y, self.each_len))
        random.shuffle(l)
        self.x, self.y, self.each_len = zip(*l)
        for i in range(len(self.x)):
            print(self.x[i].shape, self.y[i].shape, self.each_len)

train_gen = DataGenerator(train_images, train_labels, each_len, 256)

history = model.fit(train_gen, epochs=10)
#history = model.fit(train_images / 255.0, train_labels, epochs=10, batch_size=256)

model.save('model/cnn', save_format='tf')

