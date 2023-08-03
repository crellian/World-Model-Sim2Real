import numpy as np
import os
import torch
import cv2

from models import VAEBEV, BEVLSTM

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
lstm_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_LSTM_CARLA_RANDOM_BEV_CARLA_E2E_0.1_0.95_2_512.pt"


vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
vae.eval()
for param in vae.parameters():
    param.requires_grad = False
vae_ckpt = torch.load(vae_model_path, map_location="cpu")
vae.load_state_dict(vae_ckpt['model_state_dict'])

div_val = 255.0
for i in range(10):
    img = cv2.imread("manual_label/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=(0, 1))

    image_val = torch.tensor(img).to(device) / div_val
    recon, _, _ = vae(image_val)
    recon = recon.cpu().numpy().reshape((64, 64))

    cv2.imwrite(str(i)+".jpg", recon * 255)