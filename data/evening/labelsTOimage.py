import glob

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
#
# if __name__ == '__main__':
#     image_file_dir = './train/images/'
#     label_file_dir = './train/labels/'
#     for f in os.listdir(image_file_dir)[5:6]:
#         image = cv2.imread(image_file_dir+f)
#         plt.imshow(image)
#         plt.figure()
#         file_name = f.split('.png')
#         lable_path = label_file_dir+file_name[0]+'.txt'
#         with open(lable_path, 'r') as f:
#             for line in f.readlines():
#                 label = line.split(' ')
#                 print(label)
#                 plt.Rectangle((label[1],label[2]), label[3], label[4])
#             plt.show()
#             plt.close()


# 由原标签得出检测矩形框左上角和右下角的坐标分别为：xmin,ymin,xmax,ymax
def Xmin_Xmax_Ymin_Ymax(img_path, txt_path, line):
    """
    :param img_path: 图片文件的路径
    :param txt_path: 标签文件的路径
    :return:
    """
    img = cv2.imread(img_path)
    # 获取图片的高宽
    h, w, _ = img.shape
    contline = line.split(' ')
    if len(contline) > 1:
        # 计算框的左上角坐标和右下角坐标,使用strip将首尾空格去掉
        xmin = float((contline[1]).strip()) - float(contline[3].strip()) / 2
        xmax = float(contline[1].strip()) + float(contline[3].strip()) / 2

        ymin = float(contline[2].strip()) - float(contline[4].strip()) / 2
        ymax = float(contline[2].strip()) + float(contline[4].strip()) / 2

        # 将坐标（0-1之间的值）还原回在图片中实际的坐标位置
        xmin, xmax = w * xmin, w * xmax
        ymin, ymax = h * ymin, h * ymax

        return (contline[0], xmin, ymin, xmax, ymax)
    else:
        return (0, 0, 0, 2, 2)
    # 读取TXT文件 中的中心坐标和框大小




# 根据label坐标画出目标框
def plot_tangle():
    images = r'.\images\*.jpg'
    for img_path in glob.glob(images)[0:5]:
        print(img_path)
        if os.path.exists(img_path.replace('images', 'img_label')) == False:
            temp = img_path.replace('images', 'labels')
            temp = temp.replace('jpg', 'txt')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with open(temp, "r") as f:
                # 以空格划分
                for line in f.readlines():
                    cl, xmin, ymin, xmax, ymax = Xmin_Xmax_Ymin_Ymax(img_path, temp, line)
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            temp = img_path.replace('images', 'img_label')
            # plt.imsave(temp, img)
            plt.imshow(img)
            plt.show()


if __name__ == "__main__":
    plot_tangle()
    pass
