import cv2
import numpy as np


class Correction:
    def __init__(self):
        pass


    '''
        func: 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
        @param:
            height: 
            width:
            config: 获得相机与实际环境的参数
    '''
    def getRectifyTransform(self,height, width, config):
        # 读取内参和外参
        left_K = config.cam_matrix_left
        right_K = config.cam_matrix_right
        left_distortion = config.distortion_l
        right_distortion = config.distortion_r
        R = config.R
        T = config.T

        # 计算校正变换
        height = int(height)
        width = int(width)
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                          (width, height), R, T, alpha=0)
        map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
        return map1x, map1y, map2x, map2y, Q

    # 畸变校正和立体校正
    def rectifyImage(self, image1, image2, map1x, map1y, map2x, map2y):
        rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
        rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

        return rectifyed_img1, rectifyed_img2

    # 立体校正检验----画线
    def draw_line(self, image1, image2):
        # 建立输出图像
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1] + image2.shape[1]

        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:image1.shape[0], 0:image1.shape[1]] = image1
        output[0:image2.shape[0], image1.shape[1]:] = image2

        # 绘制等间距平行线
        line_interval = 50  # 直线间隔：50
        for k in range(height // line_interval):
            cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0),
                     thickness=2, lineType=cv2.LINE_AA)

        return output

from util.distance.setcameraconfig import cameraConfig

if __name__ == '__main__':
    s = cameraConfig()
    s.setMiddleComfig()
    cor = Correction()
    map1x, map1y, map2x, map2y, Q = cor.getRectifyTransform(3000,3000,s)
    image_left= cv2.imread('../../data/distance/left/1.jpg')
    image_right= cv2.imread('../../data/distance/right/1.jpg')
    rectifyed_img1, rectifyed_img2 = cor.rectifyImage(image_left, image_right, map1x, map1y, map2x, map2y)
    output = cor.draw_line(rectifyed_img1, rectifyed_img2)
    target_size = (1200,600)
    output = cv2.resize(output, target_size)
    print(output.shape)
    cv2.imshow('test', output)
    cv2.waitKey(0)
    cv2.destroyWindow('test')