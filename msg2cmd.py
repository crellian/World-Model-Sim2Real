import rospy
from sensor_msgs.msg import CompressedImage
from carla_navigation.msg import TimedTwist
import message_filters

import numpy as np

import torch
import tensorflow as tf
from cv_bridge import CvBridge
import cv2

from models import VAEBEV, StateActionLSTM

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")


vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
lstm_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_LSTM_CARLA_RANDOM_BEV_CARLA_E2E_0.1_0.95_2_512.pt"
cnn_model_path = "/lab/kiran/img2cmd_data/model/dr"


action = None

def action_callback(action_msg):
    global action
    action = action_msg

def image_callback(img_msg):
    global action
    if action is None:
        return

    np_arr = np.fromstring(img_msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    cv2.imshow("2.jpg", cv_image)
    cv2.waitKey(1)
    RGB_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    RGB_img = cv2.resize(RGB_img, (84, 84), interpolation=cv2.INTER_LINEAR)
    RGB_img = np.expand_dims(RGB_img, axis=0) / 255.0

    probs = cnn.predict(RGB_img, verbose=0)

    id = np.argmax(probs)

    z_prev = latent_cls[id]

    a = [action.twist.linear.x, action.twist.angular.y]
    z_pred = bev_lstm(torch.tensor([[a]]).to(device), z_prev)

    r = torch.reshape(vae.recon(z_pred) * 255,  (64, 64)).cpu().numpy()


    cv2.imshow("1.jpg", r)
    cv2.waitKey(1)


if __name__ == '__main__':
    vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    bev_lstm = StateActionLSTM(latent_size=32, action_size=2, hidden_size=32, batch_size=1, num_layers=1,
                         vae=vae).to(device)
    bev_lstm.eval()
    bev_lstm.init_hs()
    checkpoint = torch.load(lstm_model_path, map_location="cpu")
    bev_lstm.load_state_dict(checkpoint['model_state_dict'])
    for param in bev_lstm.parameters():
        param.requires_grad = False

    vae_ckpt = torch.load(vae_model_path, map_location="cpu")
    bev_lstm.vae.load_state_dict(vae_ckpt['model_state_dict'])


    latent_cls = []   # contains representations of 10 bev classes
    div_val = 255.0
    for i in range(10):
        img = cv2.imread("manual_label/"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=(0, 1))

        image_val = torch.tensor(img).to(device) / div_val
        z, _, _ = bev_lstm.encode(image_val)
        latent_cls.append(z)

    cnn = tf.keras.models.load_model(cnn_model_path)
#pub = rospy.Publisher('cmd', Int16, queue_size=10)

    rospy.init_node('cmd_pub', anonymous=True)
    bridge = CvBridge()

    image_sub = rospy.Subscriber(
        "image", CompressedImage, image_callback)
    action_sub = rospy.Subscriber(
        "action", TimedTwist, action_callback)

    rospy.spin()

