import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models, Model
from tensorflow.keras import Input
import os
import numpy as np
import random
import bisect
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet50

root_dir = "/home/tmp/kiran/random_bev_carla/rgb_bev"

baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(84, 84, 3)))

headModel = baseModel.output
headModel = layers.AveragePooling2D(pool_size=(3, 3))(headModel)
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(256, activation="relu")(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(30, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)


train_images = []
train_labels = []
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
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.each_len = self.cal_len()

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

    def cal_len(self):
        each_len = []
        for i in range(len(self.y)):
            if len(each_len) == 0:
                each_len.append(self.y[i].shape[0])
            else:
                each_len.append(self.y[i].shape[0] + each_len[-1])
        return each_len

    def on_epoch_end(self):
        l = list(zip(self.x, self.y))
        random.shuffle(l)
        self.x, self.y = zip(*l)
        self.each_len = self.cal_len()

model.compile(optimizer='adam',
                      loss="sparse_categorical_crossentropy",
                                    metrics=['accuracy'])


train_gen = DataGenerator(train_images, train_labels, 256)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
              filepath='model',
               save_freq=10000,)

history = model.fit(train_gen, epochs=100, workers=8, callbacks=[cp_callback])

model.save('model/cnn', save_format='tf')

