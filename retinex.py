import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

from util.rtmodel import DecomNet, RelightNet,RetinexNet


def show_image(image):
    # 对图像噪点进行修正并统计时间
    # b, g, r = cv2.split(image)
    # image = cv2.merge([r, g, b])
    # image = cv2.fastNlMeansDenoisingColored( image, None, 10, 10, 7, 21)
    cv2.imshow('image', image)
    cv2.waitKey(0)


# image = cv2.imread('./data/evening/images/00329.jpg')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RetinexNet = RetinexNet().to(device)
# RetinexNet.predict_load()
# image_high = RetinexNet.predict(image, True)
# show_image(image_high)


source = './data/evening4/images/'
save_dir = './data/evening5/images/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RetinexNet = RetinexNet().to(device)
RetinexNet.predict_load()


for file_name in os.listdir(source):
    load_path = source+file_name
    save_path = save_dir+file_name
    image = cv2.imread(load_path)
    image_high = RetinexNet.predict(image,False)
    cv2.imwrite(save_path, image_high)
