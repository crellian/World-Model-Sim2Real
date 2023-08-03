import numpy as np
import os
import pickle
import torch
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from models import VAEBEV

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")
vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"

if __name__ == '__main__':
    vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    vae_ckpt = torch.load(vae_model_path, map_location="cpu")
    vae.load_state_dict(vae_ckpt['model_state_dict'])

    div_val = 255.0
    root_dir = "/home/tmp/kiran/random_bev_carla/rgb_bev"
    latents = None
    for root, subdirs, files in os.walk(root_dir):
        for f in files:
            if 'observation.npy' in f and 'rgb' not in f:
                bevs = np.load(os.path.join(root, f))[:, :, :, 0]
                bevs = np.expand_dims(bevs, axis=1)
                image_val = torch.tensor(bevs).to(device) / div_val
                mean, std = torch.mean(image_val), torch.std(image_val)
                image_val = (image_val - mean) / std

#    for i in range(100):
#        img = cv2.imread("/home2/offline_carla/dataset_test/test/"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE)
#        img = np.expand_dims(img, axis=(0, 1))
#        image_val = torch.tensor(img).to(device) / div_val

                step = int(image_val.shape[0] / 10)

                for i in range(10):
                    _, mu, logvar = vae(image_val[i*step:(i+1)*step])
                    z = vae.reparameterize(mu, logvar).cpu()

                    latents = z.numpy() if latents is None else np.concatenate((latents, z), axis=0)

                print(latents.shape)


    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    k = 10
    kmeans = KMeans(n_clusters = k).fit(latents)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    with open("kmeans.pkl", "wb") as f:
            pickle.dump(kmeans, f)
#with open("kmeans.pkl", "rb") as f:
    #    kmeans = pickle.load(f)
    c_imgs = vae.recon(torch.tensor(centroids).to(device)) # center images
    for i, c in enumerate(c_imgs):
        c = torch.permute(c, (1, 2, 0))
        cv2.imwrite("centers/"+str(k)+"_"+str(i)+".jpg", c.cpu().numpy()*255)


    #for i in range(10):
    #    print(pred_clusters[i*10:i*10+10])
    j = 0
    for root, subdirs, files in os.walk(root_dir):
        for f in files:
            if 'observation.npy' in f:
                latents = None
                bevs = np.load(os.path.join(root, f))[:, :, :, 0]
                bevs_c = np.expand_dims(bevs, axis=1)
                image_val = torch.tensor(bevs_c).to(device) / div_val
                n = int(image_val.shape[0] / 10000)
                print(image_val.shape)

                for i in range(n):
                    _, mu, logvar = vae(image_val[i*10000:(i+1)*10000])
                    z = vae.reparameterize(mu, logvar).cpu()

                    latents = z.numpy() if latents is None else np.concatenate((latents, z), axis=0)

                _, mu, logvar = vae(image_val[n*10000:])
                z = vae.reparameterize(mu, logvar).cpu()

                latents = np.concatenate((latents, z), axis=0)

                print(latents.shape)

                pred_clusters = kmeans.predict(latents)  # labels

                print(pred_clusters.shape)

                np.save(os.path.join(root, "label.npy"), pred_clusters)
                for k, im in enumerate(bevs[:100]):
                    cv2.imwrite("samples/"+str(j)+"_"+str(k)+"_"+str(pred_clusters[k])+".jpg", im)
                j+=1