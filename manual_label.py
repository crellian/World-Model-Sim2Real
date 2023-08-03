import numpy as np
import os

import torch
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from models import VAEBEV

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")
vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
root_dir = "/home2/val_bev_carla/rgb_bev"
if __name__ == '__main__':
    vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    vae_ckpt = torch.load(vae_model_path, map_location="cpu")
    vae.load_state_dict(vae_ckpt['model_state_dict'])

    div_val = 255.0

    cls_mu = []
    cls_logvar = []
    anchors = []
    for i in range(10):
        img = cv2.imread("manual_label/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        anchors.append(img)
        img = np.expand_dims(img, axis=(0, 1))

        image_val = torch.tensor(img).to(device) / div_val
        _, mu, logvar = vae(image_val)
        cls_mu.append(mu[0].cpu().numpy())
        cls_logvar.append(logvar[0].cpu().numpy())

    cls_mu = np.array(cls_mu)
    cls_logvar = np.array(cls_logvar)

    for root, subdirs, files in os.walk(root_dir):
        for f in files:
            if 'observation.npy' in f and 'rgb' not in f:
                print(root)
                label = []
                bevs = np.load(os.path.join(root, f))[:, :, :, 0]

                for i in range(len(bevs)):
                    bevs_c = np.expand_dims(bevs[i], axis=(0, 1))
                    image_val = torch.tensor(bevs_c).to(device) / div_val

                    _, mu, logvar = vae(image_val)
                    d_mu = np.linalg.norm(cls_mu - mu[0].cpu().numpy(), axis=1)
                    d_logvar = np.linalg.norm(cls_logvar - logvar[0].cpu().numpy(), axis=1)

                    idx = np.argmin(d_mu + d_logvar)

                    label.append(idx)

                label = np.array(label)
                print(label.shape)
                np.save(os.path.join(root, "label.npy"), label)





