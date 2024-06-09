import os
import time

import cv2
import numpy as np

class Retinex(object):
    # 对图像进行单尺度 Retinex 处理
    def single_scale_retinex(self,img, sigma):
        # 通过高斯混合
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        return retinex

    # 对图像进行多尺度 Retinex 处理
    def multi_scale_retinex(self, img, sigma_list):
        retinex = np.zeros_like(img)
        for sigma in sigma_list:
            retinex += self.single_scale_retinex(img, sigma)
        retinex = retinex / len(sigma_list)
        return retinex

    # 进行颜色恢复
    def color_restoration(self,img, alpha, beta):
        img_sum = np.sum(img, axis=2, keepdims=True)
        color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
        return color_restoration

    # 图像增强主函数，包括图像增强和颜色恢复 使用多尺度Retinex MSR
    def retinex_process(self,img, sigma_list, G, b, alpha, beta):
        img = np.float64(img) + 1.0
        img_retinex = self.multi_scale_retinex(img, sigma_list)
        img_color = self.color_restoration(img, alpha, beta)
        img_retinex = G * (img_retinex * img_color + b)
        # 将像素值限制在范围内
        for i in range(img_retinex.shape[2]):
            img_retinex[:, :, i] = np.clip(img_retinex[:, :, i], 0, 255)
        img_retinex = np.uint8(img_retinex)
        return img_retinex


def meanFiltering1(img, size):  # img输入，size均值滤波器的尺寸，>=3，且必须为奇数
    num = int((size - 1) / 2)  # 输入图像需要填充的尺寸
    img = cv2.copyMakeBorder(img, num, num, num, num, cv2.BORDER_REPLICATE)  # 对传入的图像进行扩充，尺寸为num
    h1, w1 = img.shape[0:2]
    # 高斯滤波
    img1 = np.zeros((h1, w1, 3), dtype="uint8")  # 定义空白图像，用于输出中值滤波后的结果
    for i in range(num, h1 - num):  # 对扩充图像中的原图进行遍历
        for j in range(num, w1 - num):
            sum = 0
            sum1 = 0
            sum2 = 0
            for k in range(i - num, i + num + 1):  # 求中心像素周围size*size区域内的像素的平均值
                for l in range(j - num, j + num + 1):
                    sum = sum + img[k, l][0]  # B通道
                    sum1 = sum1 + img[k, l][1]  # G通道
                    sum2 = sum2 + img[k, l][2]  # R通道
            sum = sum / (size ** 2)  # 除以核尺寸的平方
            sum1 = sum1 / (size ** 2)
            sum2 = sum2 / (size ** 2)
            img1[i, j] = [sum, sum1, sum2]  # 复制给空白图像
    return img1
    # img1 = img1[(0 + num):(h1 - num), (0 + num):(h1 - num)]  # 从滤波图像中裁剪出原图像


if __name__ == "__main__":
    # 尺度列表
    sigma_list = [15, 80, 250]
    # 增益参数
    G = 5.0
    # 偏置参数
    b = 25.0
    # 颜色恢复参数
    alpha = 125.0
    # 颜色恢复参数
    beta = 46.0

    # source = './eveningdataset/images/'
    # save_dir = './eveningdataset1/images/'
    # retinex = Retinex()

    # 对retinex的原理进行高斯滤波
    source1 =  './eveningdataset1/images/'
    save_dir2 = './eveningdataset4/images/'
    for file_name in os.listdir(source1):
        load_path = source1 + file_name
        save_path = save_dir2 + file_name
        img = cv2.imread(load_path)
        blur_img = cv2.GaussianBlur(img, (5,5), 0)
        cv2.imwrite(save_path, blur_img)
        print(file_name,"处理完毕", sep='')




    # # img = cv2.imread('./data/evening/images/00328.jpg')
    # for file_name in os.listdir(source):
    #     load_path = source + file_name
    #     save_path = save_dir + file_name
    #     img = cv2.imread(load_path)
    #     # start_time = time.time()
    #     img_retinex = retinex.retinex_process(img, sigma_list, G, b, alpha, beta)
    #     cv2.imwrite(save_path, img_retinex)
    #     print(file_name,"处理完毕", sep='')
    #     # 时间2s
    #     # print(time.time()-start_time)
    #     # img_retinex = meanFiltering1(img_retinex, 5)  # 均值滤波 噪声过滤时间长
    #     # cv2.imshow('retinex', img_retinex)
    #     # cv2.imshow('img', img)
    #     # cv2.waitKey(0)

    # file = './data/evening2/images/00328.jpg'
    # img = cv2.imread(file)
    # cv2.imshow('img',img)

    # 算数平均滤波
    # rec_img = meanFiltering1(img, 3)
    # cv2.imshow('average', rec_img)

    # 反谐波平均滤波
    # img = img/255
    # img1 = np.transpose(img, (2, 0, 1))  # 转换成[channel,H,W]形式
    # m = 3  # 定义滤波核大小
    # n = 3
    # Q = 0.1
    # rec_img = np.zeros((img1.shape[0], img1.shape[1] - m + 1, img1.shape[2] - n + 1))
    # for channel in range(rec_img.shape[0]):
    #     for i in range(rec_img[channel].shape[0]):
    #         for j in range(rec_img[channel].shape[1]):
    #             rec_img[channel][i, j] = ((np.power(img1[channel][i:i + m, j:j + n], Q + 1)).sum()) / (
    #                 (np.power(img1[channel][i:i + m, j:j + n], Q)).sum())
    # rec_img = np.transpose(rec_img, (1, 2, 0))
    # cv2.imshow('average', rec_img)

    # 修正阿尔法均值滤波 效果不错
    # img = img /255
    # img1 = np.transpose(img, (2, 0, 1))  # 转换成[channel,H,W]形式
    # m = 3  # 定义滤波核大小
    # n = 3
    # d = 4  # d取偶数
    # rec_img = np.zeros((img1.shape[0], img1.shape[1] - m + 1, img1.shape[2] - n + 1))
    # for channel in range(rec_img.shape[0]):
    #     for i in range(rec_img[channel].shape[0]):
    #         for j in range(rec_img[channel].shape[1]):
    #             img2 = np.sort(np.ravel(img1[channel][i:i + m, j:j + n]))  # np.ravel():多维数组变成一维数组
    #             rec_img[channel][i, j] = (img2[int(d / 2):-int(d / 2)].sum()) * (1 / (m * n - d))
    # rec_img = np.transpose(rec_img, (1, 2, 0))
    # cv2.imshow('alpha average', rec_img)


    # img = img / 255
    #
    # blurred_image = cv2.GaussianBlur(img, (3, 3), 0)
    # cv2.imshow('blur', blurred_image)
    # cv2.waitKey(0)

