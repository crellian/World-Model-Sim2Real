import numpy as np
import cv2

task = "Town10_1"

#rgb = np.load("/home2/random_bev_carla/rgb_bev/"+task+"/5/0/observation_rgb.npy")

# ground truth bev
bev = np.load("/home2/random_bev_carla/rgb_bev/"+task+"/5/0/observation.npy")[:, :, :, 0]

# ground truth label -- obtained by minimum distance
label = np.load("/home2/random_bev_carla/rgb_bev/"+task+"/5/0/label.npy")

anchors = []
for i in range(10):
    img = cv2.imread("manual_label/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
    anchors.append(img)

for i in range(len(bev)):
    #im = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR)
    #cv2.imshow("rgb", im)
    cv2.imshow("bev", bev[i])
    cv2.imshow("anchor", anchors[label[i]])
    cv2.waitKey(100)