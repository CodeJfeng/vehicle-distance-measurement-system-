import os

import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)

# # 读取图像
# original = cv2.imread("img.png", 1)
#
# # gamma矫正
# gamma = 2.0 # 矫正系数：越大越亮、越小越暗 1.8
# adjusted = adjust_gamma(original, gamma=gamma)
# cv2.imshow("original.jpg", original)
# cv2.imshow("result.jpg", adjusted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#




gamma = 1.6 # 矫正系数：越大越亮、越小越暗 1.8

# source1 =  '../data/evening/images/'
# save_dir2 = '../data/evening4/images/'
# for file_name in os.listdir(source1):
#     load_path = source1 + file_name
#     save_path = save_dir2 + file_name
#     img = cv2.imread(load_path)
#     blur_img = adjust_gamma(img, gamma=gamma)
#     blur_img = cv2.GaussianBlur(blur_img, (3, 3), 0)
#     cv2.imwrite(save_path, blur_img)
#     print(file_name,"处理完毕", sep='')
