import os
import sys
import time
import argparse
import random
import bisect

os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

devices = tf.config.experimental.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.experimental.set_visible_devices(devices[0], 'GPU')

import tensorflow_probability as tfp
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import datasets, layers, models, Model
from tensorflow.keras import Input

from ppo import PPO

from PIL import Image, ImageEnhance
import cv2


def normalize_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    return frame


def transform_frame(frame, transform_params):
    frame = Image.fromarray(frame)
    brightness = ImageEnhance.Brightness(frame)
    contrast = ImageEnhance.Contrast(frame)
    hue = ImageEnhance.Color(frame)
    frame = brightness.enhance(transform_params[0])
    frame = contrast.enhance(transform_params[1])
    frame = hue.enhance(transform_params[2])
    frame = np.asarray(frame)
    return frame


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, transform_params=None):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.each_len = self.cal_len()
        self.transform_params = transform_params

    def __len__(self):
        return int(np.ceil(self.each_len[-1] / float(self.batch_size)))

    def __getitem__(self, idx):
        idx = idx * self.batch_size
        file_ind = bisect.bisect_right(self.each_len, idx)
        if file_ind == 0:
            im_ind = idx
        else:
            im_ind = idx - self.each_len[file_ind - 1]
        # print(idx, file_ind, im_ind, self.each_len)

        batch_x = self.x[file_ind][im_ind:im_ind + self.batch_size]
        batch_y = self.y[file_ind][im_ind:im_ind + self.batch_size]

        if self.transform_params is not None:
            batch_x = transform_frame(batch_x, self.transform_params)
        batch_x = normalize_frame(batch_x)

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


class DRParameters:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.brightness = tfp.distributions.Normal(
            tf.Variable(self.hyper_params['brightness']['mu'], name='b_mu'),
            tf.Variable(self.hyper_params['brightness']['sigma'], name='b_sigma')
        )
        self.contrast = tfp.distributions.Normal(
            tf.Variable(self.hyper_params['contrast']['mu'], name='c_mu'),
            tf.Variable(self.hyper_params['contrast']['sigma'], name='c_sigma'),
        )
        self.hue = tfp.distributions.Normal(
            tf.Variable(self.hyper_params['hue']['mu'], name='h_mu'),
            tf.Variable(self.hyper_params['hue']['sigma'], name='h_sigma'),
        )

    def sample(self):
        return [self.brightness.sample(), self.contrast.sample(), self.hue.sample()]


class DomainRandomizer:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.init_cnn()
        self.params = DRParameters(self.hyper_params['dr'])
        self.trainable_params = {
            'b': self.params.brightness.trainable_variables,
            'c': self.params.contrast.trainable_variables,
            'h': self.params.hue.trainable_variables,
        }
        self.best_eval_loss = np.inf
        self.epochs = self.hyper_params['dr']['epochs']
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.hyper_params['dr']['learning_rate'],
                                                name='dr_optimizer')

        self.optimizer.apply_gradients(zip([0.0, 0.0], self.trainable_params['b']))
        self.optimizer.apply_gradients(zip([0.0, 0.0], self.trainable_params['c']))
        self.optimizer.apply_gradients(zip([0.0, 0.0], self.trainable_params['h']))

    def init_dataset(self, root_dir, in_source_domain=True):
        images = []
        labels = []
        if in_source_domain:
            for root, subdirs, files in os.walk(root_dir):
                for f in files:
                    if 'observation_rgb.npy' in f:
                        print(root)
                        rgb = np.load(os.path.join(root, f), mmap_mode='r')
                        print(rgb.shape)
                        images.append(rgb)

                    elif 'label.npy' in f:
                        l = np.load(os.path.join(root, f))
                        labels.append(l)
        else:
            for root, subdirs, files in os.walk(root_dir):
                img = []
                for f in files:
                    if '.jpg' in f:
                        rgb = cv2.imread(os.path.join(root, f))
                        img.append(rgb)
                if len(files) > 0:
                    images.append(np.array(img))
                    labels.append(np.ones(len(files)) * int(root[-1]))
                    print(root)
                    print(images[-1].shape)
        return images, labels

    def init_cnn(self):
        baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(84, 84, 3)))

        headModel = baseModel.output
        headModel = layers.AveragePooling2D(pool_size=(3, 3))(headModel)
        headModel = layers.Flatten(name="flatten")(headModel)
        headModel = layers.Dense(256, activation="relu")(headModel)
        headModel = layers.Dropout(0.5)(headModel)
        headModel = layers.Dense(30, activation="softmax")(headModel)

        self.cnn = Model(inputs=baseModel.input, outputs=headModel)

        self.cnn.compile(optimizer='adam',
                         loss="sparse_categorical_crossentropy",
                         metrics=['accuracy'])

    def train(self, transform_params):
        # self.model.reset_episode_idx()
        epochs = self.hyper_params['model']['epochs']
        batch_size = self.hyper_params['model']['batch_size']

        # prepare dataset
        images, labels = self.init_dataset("/home/tmp/kiran/random_bev_carla/rgb_bev")
        train_gen = DataGenerator(images, labels, batch_size, transform_params)

        # train model
        history = self.cnn.fit(train_gen, epochs=epochs, workers=8)

        return history['loss']

    def eval(self, transform_params, in_source_domain=True):
        batch_size = self.hyper_params['model']['batch_size']
        # prepare dataset
        if in_source_domain:
            images, labels = self.init_dataset("/home/tmp/kiran/val_bev_carla/rgb_bev")
            val_gen = DataGenerator(images, labels, batch_size, transform_params)
        else:
            images, labels = self.init_dataset("/home/tmp/kiran/USC_GStView", False)
            val_gen = DataGenerator(images, labels, batch_size)
        # evaluate model
        loss = self.cnn.evaluate(val_gen)

        return loss

    def run(self):
        # self.model.sess.run(tf.variables_initializer(self.optimizer.variables()))
        for idx in range(self.epochs):
            print(f'DR Epoch {idx}')
            transform_params = self.params.sample()
            print(self.train(transform_params))
            source_loss = self.eval(transform_params)
            target_loss = self.eval(transform_params, in_source_domain=False)
            print("source_loss: ", source_loss)
            print("target_loss: ", target_loss)

            if target_loss < self.best_eval_loss:
                print("Parameters update: ", transform_params)
                self.cnn.save('../img2cmd_data/model/cnn', save_format='tf')
                self.best_eval_loss = target_loss

            self.transfer_loss = (target_loss - source_loss)

            with tf.GradientTape() as tape:
                bl = -tf.math.log(self.params.brightness.prob(transform_params[0]) * self.transfer_loss)
                cl = -tf.math.log(self.params.contrast.prob(transform_params[1]) * self.transfer_loss)
                hl = -tf.math.log(self.params.hue.prob(transform_params[2]) * self.transfer_loss)

                bg = tape.gradients(bl, self.trainable_params['b'])
                cg = tape.gradients(cl, self.trainable_params['c'])
                hg = tape.gradients(hl, self.trainable_params['h'])

            self.optimizer.apply_gradients(zip(bg, self.trainable_params['b'])),
            self.optimizer.apply_gradients(zip(cg, self.trainable_params['c'])),
            self.optimizer.apply_gradients(zip(hg, self.trainable_params['h'])),

        print('Done...')


def init_hyper_params():
    parser = argparse.ArgumentParser(description="Domain Randomization (sim2sim)")

    # DR hyper parameters
    parser.add_argument("--dr_learning_rate", type=float, default=1e-2, help="DR learning rate")
    parser.add_argument("--dr_num_epochs", type=int, default=500, help="DR number of epochs")
    parser.add_argument("--brightness_mean", type=float, default=1.5, help="Initial Distribution Mean (Brightness)")
    parser.add_argument("--brightness_std", type=float, default=0.5, help="Initial Distribution Std Dev (Brightness)")
    parser.add_argument("--contrast_mean", type=float, default=1.5, help="Initial Distribution Mean (Contrast)")
    parser.add_argument("--contrast_std", type=float, default=0.5, help="Initial Distribution Std Dev (Contrast)")
    parser.add_argument("--hue_mean", type=float, default=1.5, help="Initial Distribution Mean (Hue)")
    parser.add_argument("--hue_std", type=float, default=0.5, help="Initial Distribution Std Dev (Hue)")

    # PPO hyper parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="Per-episode exponential learning rate decay")
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon")
    parser.add_argument("--initial_std", type=float, default=1.0,
                        help="Initial value of the std used in the gaussian policy")
    parser.add_argument("--value_scale", type=float, default=1.0, help="Value loss scale factor")
    parser.add_argument("--entropy_scale", type=float, default=0.01, help="Entropy loss scale factor")

    # VAE parameters
    parser.add_argument("--vae_model", type=str,
                        default="seg_bce_cnn_zdim64_beta1_kl_tolerance0.0_data",
                        help="Trained VAE model to load")
    parser.add_argument("--vae_model_type", type=str, default='cnn', help="VAE model type (\"cnn\" or \"mlp\")")
    parser.add_argument("--vae_z_dim", type=int, default=64, help="Size of VAE bottleneck")
    parser.add_argument("-use_vae", action="store_true", help="If True, use vae, else mobilenet v2")

    # General hyper parameters
    parser.add_argument("--discount_factor", type=float, default=0.99, help="GAE discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--horizon", type=int, default=128, help="Number of steps to simulate per training step")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of PPO training epochs per traning step")
    parser.add_argument("--batch_size", type=int, default=256, help="Epoch batch size")
    parser.add_argument("--num_episodes", type=int, default=2500,
                        help="Number of episodes to train for (0 or less trains forever)")

    # Common Environment settings
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to connect to")
    parser.add_argument("--fps", type=int, default=15, help="Set this to the FPS of the environment")
    parser.add_argument("--action_smoothing", type=float, default=0.3, help="Action smoothing factor")
    parser.add_argument("--reward_fn", type=str,
                        default="reward_speed_centering_angle_multiply",
                        help="Reward function to use. See reward_functions.py for more info.")

    # Carla Settings
    parser.add_argument("--synchronous", type=int, default=True,
                        help="Set this to True when running in a synchronous environment")
    parser.add_argument("-weather", action="store_true", help="If True, use all weather presets to train every episode")

    # AirSim Settings
    parser.add_argument("--route_file", type=str, default="./AirSimEnv/routes/dr-test-02.txt",
                        help="Route to use in AirSim")

    parser.add_argument("--model_name", type=str, default=f"dr-model-{int(time.time())}",
                        help="Name of the model to train. Output written to models/model_name")

    params = vars(parser.parse_args())

    return {
        'model': {
            'ppo': {
                'learning_rate': params['learning_rate'],
                'lr_decay': params['lr_decay'],
                'epsilon': params['ppo_epsilon'],
                'initial_std': params['initial_std'],
                'value_scale': params['value_scale'],
                'entropy_scale': params['entropy_scale'],
                'model_name': params['model_name']
            },
            'vae': {
                'model_name': params['vae_model'],
                'model_type': params['vae_model_type'],
                'z_dim': params['vae_z_dim']
            },
            'horizon': params['horizon'],
            'epochs': params['num_epochs'],
            'episodes': params['num_episodes'],
            'batch_size': params['batch_size'],
            'gae_lambda': params['gae_lambda'],
            'discount_factor': params['discount_factor'],
            'use_vae': params['use_vae']
        },
        'env': {
            'common': {
                'host': params['host'],
                'fps': params['fps'],
                'action_smoothing': params['action_smoothing'],
                'reward_fn': params['reward_fn'],
                'obs_res': (160, 80) if params['use_vae'] else (160, 160)
            },
            'source': {
                'synchronous': params['synchronous'],
                'start_carla': False,
                'weather': params['weather']
            },
            'target': {
                'route_file': params['route_file']
            }
        },
        'dr': {
            'brightness': {
                'mu': params['brightness_mean'],
                'sigma': params['brightness_std']
            },
            'contrast': {
                'mu': params['contrast_mean'],
                'sigma': params['contrast_std']
            },
            'hue': {
                'mu': params['hue_mean'],
                'sigma': params['hue_std']
            },
            'epochs': params['dr_num_epochs'],
            'learning_rate': params['dr_learning_rate']
        }
    }


if __name__ == '__main__':
    hyper_params = init_hyper_params()
    dr = DomainRandomizer(hyper_params)
    try:
        dr.run()
    except Exception as e:
        raise e
