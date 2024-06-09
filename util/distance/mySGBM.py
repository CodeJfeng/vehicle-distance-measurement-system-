import time

import cv2
import numpy as np

from util.distance.Horizontal import Correction
from util.distance.setcameraconfig import cameraConfig


class mySGBM:
    '''
        func: 匹配和寻找视差
        method: SGBM 和 滤波平滑
    '''

    def __init__(self):
        pass

    # 视差计算
    def stereoMatchSGBM(self, left_image, right_image, down_scale=False):
        # SGBM匹配参数设置
        if left_image.ndim == 2:
            img_channels = 1
        else:
            img_channels = 3
        blockSize = 3
        paraml = {'minDisparity': 0,
                  'numDisparities': 128,
                  'blockSize': blockSize,
                  'P1': 8 * img_channels * blockSize ** 2,
                  'P2': 32 * img_channels * blockSize ** 2,
                  'disp12MaxDiff': 1,
                  'preFilterCap': 63,
                  'uniquenessRatio': 15,
                  'speckleWindowSize': 100,
                  'speckleRange': 1,
                  'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                  }

        # 构建SGBM对象
        left_matcher = cv2.StereoSGBM_create(**paraml)
        paramr = paraml
        paramr['minDisparity'] = -paraml['numDisparities']
        right_matcher = cv2.StereoSGBM_create(**paramr)

        # 计算视差图
        size = (left_image.shape[1], left_image.shape[0])
        if down_scale == False:
            disparity_left = left_matcher.compute(left_image, right_image)
            disparity_right = right_matcher.compute(right_image, left_image)

        else:
            left_image_down = cv2.pyrDown(left_image)
            right_image_down = cv2.pyrDown(right_image)
            factor = left_image.shape[1] / left_image_down.shape[1]

            disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
            disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
            disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
            disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
            disparity_left = factor * disparity_left
            disparity_right = factor * disparity_right

        # 真实视差（因为SGBM算法得到的视差是×16的）
        trueDisp_left = disparity_left.astype(np.float32) / 16.
        trueDisp_right = disparity_right.astype(np.float32) / 16.

        return trueDisp_left, trueDisp_right

    '''
        func: 构建云图，暂时省略
    '''

    def hw3ToN3(points):
        height, width = points.shape[0:2]

        points_1 = points[:, :, 0].reshape(height * width, 1)
        points_2 = points[:, :, 1].reshape(height * width, 1)
        points_3 = points[:, :, 2].reshape(height * width, 1)

        points_ = np.hstack((points_1, points_2, points_3))

        return points_

    def DepthColor2Cloud(points_3d, colors):
        rows, cols = points_3d.shape[0:2]
        size = rows * cols

        points_ = hw3ToN3(points_3d)
        colors_ = hw3ToN3(colors).astype(np.int64)

        # 颜色信息
        blue = colors_[:, 0].reshape(size, 1)
        green = colors_[:, 1].reshape(size, 1)
        red = colors_[:, 2].reshape(size, 1)

        rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

        # 将坐标+颜色叠加为点云数组
        pointcloud = np.hstack((points_, rgb)).astype(np.float32)

        # 删掉一些不合适的点
        X = pointcloud[:, 0]
        Y = -pointcloud[:, 1]
        Z = -pointcloud[:, 2]

        remove_idx1 = np.where(Z <= 0)
        remove_idx2 = np.where(Z > 15000)
        remove_idx3 = np.where(X > 10000)
        remove_idx4 = np.where(X < -10000)
        remove_idx5 = np.where(Y > 10000)
        remove_idx6 = np.where(Y < -10000)
        remove_idx = np.hstack(
            (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

        pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

        return pointcloud_1



if __name__ == '__main__':
    s = cameraConfig()
    # s.setMiddleComfig()
    cor = Correction()
    map1x, map1y, map2x, map2y, Q = cor.getRectifyTransform(320,480,s)
    # image_left= cv2.imread('../../data/distance/left/test_01.jpg')
    # image_right= cv2.imread('../../data/distance/right/test_02.jpg')
    img = cv2.imread("../../data/images/05141803_01.jpg")
    h, w, _ = img.shape
    half_w = int(w / 2)
    image_left = img[:h, :half_w]
    image_right = img[:h, half_w:]
    cv2.imshow("left", image_left)
    start_time = time.time()
    rectifyed_img1, rectifyed_img2 = cor.rectifyImage(image_left, image_right, map1x, map1y, map2x, map2y)
    my = mySGBM()
    disp, _  = my.stereoMatchSGBM(rectifyed_img1, rectifyed_img2, True)
    # target_size = (1200, 600)
    print(time.time() - start_time)
    # print(_)
    # disp = cv2.resize(disp, target_size)
    # cv2.imshow('test', disp)
    # cv2.waitKey(0)
    # cv2.destroyWindow('test')
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # 可以使用上文的stereo_config.py给出的参数


    # points_3d = points_3d

    # 鼠标点击事件
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('点 (%d, %d) 的三维坐标 (%f, %f, %f)' % (x, y, points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2]))
            dis = ((points_3d[y, x, 0] ** 2 + points_3d[y, x, 1] ** 2 + points_3d[y, x, 2] ** 2) ** 0.5) / 1000 # 计算三位空间的真实距离
            print('点 (%d, %d) 距离左摄像头的相对距离为 %0.3f m' % (x, y, dis))

    # 显示图片
    cv2.namedWindow("disparity", 0)
    cv2.imshow("disparity", disp)
    cv2.setMouseCallback("disparity", onMouse, 0)

    # 构建点云--Point_XYZRGBA格式
    # pointcloud = my.DepthColor2Cloud(points_3d)
    # # 显示点云
    # view_cloud(pointcloud)

    cv2.waitKey(0)
    cv2.destroyAllWindows()