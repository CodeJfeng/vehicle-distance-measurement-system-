import cv2
import numpy as np
import torch.nn
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])


# 创建一个高斯噪声的图像
image = cv2.imread('../../data/images/240517092951.jpg')
image = cv2.pyrDown(image)
image = cv2.pyrDown(image)
image = torch.nn.Upsample(image, mode='nearest')
print(type(image))
image = transform(image)

# 保存下采样后的图像
# np.save('downsampled_image', image)
# print(r)
cv2.imshow('test', image)
cv2.waitKey(0)
cv2.destroyWindow()