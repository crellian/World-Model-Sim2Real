import numpy as np
import cv2

rgb = np.load("/home2/random_bev_carla/rgb_bev/Town10_0/5/0/observation_rgb.npy")[70000:]

# ground truth bev
bev = np.load("/home2/random_bev_carla/rgb_bev/Town10_0/5/0/observation.npy")[70000:, :, :, 0]

for i in range(len(rgb)):
    cv2.imshow("rgb.jpg", rgb[i])
    cv2.imshow("output.jpg", bev[i])
    cv2.waitKey(1)