import time

from util.Gammer import adjust_gamma
from util.colors import Colors
import cv2
import numpy as np
import torch
import torchvision


# 定义函数 速度与刹车距离定义函数
def f(x):
    return x ** 1.7 * 0.012


# 定义 宽度与高的距离公式
def line1(y):
    return -(y - 400) / 1.1333333 + 900


# 定义 宽度与高的距离公式
def line2(y):
    return (y - 400) / 1.1333333 + 1020


class Onnx_clf:
    def __init__(self, onnx: str = './yolov5/runs/train/exp14/weights/best.onnx', img_size=640,
                 classlist: list = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle', 'tricycle']) -> None:
        '''	@func: 读取onnx模型,并进行目标识别
            @para	onnx:模型路径 ./yolov5/runs/train/exp5/weights/last.onnx
                 	img_size:输出图片大小,和模型直接相关
                    classlist:类别列表
            @return: None
        '''
        self.net = cv2.dnn.readNetFromONNX(onnx)  # 读取模型
        self.img_size = img_size  # 输出图片尺寸大小
        self.classlist = classlist  # 读取类别列表
        self.pre_pixel = [1080, 865, 805, 760, 690, 655, 625, 600, 570, 550, 536, 523, 499, 480, 460, 440, 422, 402,
                          380, 360, 340, 325, 310]
        self.pre_dist = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44]
        self.classlist_han = ['汽车', '卡车', '公交车', '行人', '自行车', '摩托车', '三轮车']
        self.liangci = ['辆', '辆', '辆', '个', '辆', '辆', '辆']
        self.colors = Colors().palette

    def img_identify(self, img, ifshow=True, speed=0, conf = 0.25) -> np.ndarray:
        '''	@func: 图片识别
            @para	img: 图片路径或者图片数组
                    ifshow: 是否显示图片
            @return: 图片数组
        '''
        if type(img) == str:
            src = cv2.imread(img)
        else:
            src = img
        height, width, _ = src.shape
        _max = max(width, height)
        resized = np.zeros((_max, _max, 3), np.uint8)
        resized[0:height, 0:width] = src  # 将图片转换成正方形，防止后续图片预处理(缩放)失真
        # 图像预处理函数,缩放裁剪,交换通道  img     scale              out_size              swapRB
        blob = cv2.dnn.blobFromImage(resized, 1 / 255.0, (self.img_size, self.img_size), swapRB=True)
        prop = _max / self.img_size  # 计算缩放比例
        dst = cv2.resize(src, (round(width / prop), round(height / prop)))
        self.net.setInput(blob)  # 将图片输入到模型
        out = self.net.forward()  # 模型输出
        out = np.array(out[0])
        out = out[out[:, 4] >= conf]  # 过滤置信度低的目标
        boxes = out[:, :4]
        confidences = out[:, 4]
        class_ids = np.argmax(out[:, 5:], axis=1)
        class_scores = np.max(out[:, 5:], axis=1)
        # out2 = out[0][out[0][:][4] > 0.5]
        # for i in out[0]: # 遍历每一个框
        #     class_max_score = max(i[5:])
        #     if i[4] < 0.5 or class_max_score < 0.25: # 过滤置信度低的目标
        #         continue
        #     boxes.append(i[:4]) # 获取目标框: x,y,w,h (x,y为中心点坐标)
        #     confidences.append(i[4]) # 获取置信度
        #     class_ids.append(np.argmax(i[5:])) # 获取类别id
        #     class_scores.append(class_max_score) # 获取类别置信度
        # 原Yolo代码  i = torchvision.ops.nms(boxes候选框, scores置信度, iou_thres比率)  # NMS [tensor, tensor, 0.45 ]
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)  # 非极大值抑制, 获取的是索引
        print(('检测到{}个物体'.format(len(indexes))))
        iffall = True if len(indexes) != 0 else False
        distance_warn = f(speed)

        self.drawLine(src, 0)  # 画危险区域
        for (i, c) in zip(indexes, confidences):  # 遍历每一个目标, 绘制目标框
            box = boxes[i]
            class_id = class_ids[i]
            score = round(class_scores[i], 2)
            x1 = round((box[0] - 0.5 * box[2]) * prop)
            y1 = round((box[1] - 0.5 * box[3]) * prop)
            x2 = round((box[0] + 0.5 * box[2]) * prop)
            y2 = round((box[1] + 0.5 * box[3]) * prop)
            # 重写 cv2.矩阵
            src = np.ascontiguousarray(src)  # 将矩阵重新放在一连续的内存空间
            distance = self.mydistance(src, [x1, y1, x2, y2])
            print(self.classlist[class_id], x1, y1, x2, y2, distance)
            if distance < distance_warn and y2 > 400:  # 当距离过近且距离较低时
                d1 = line1(y2)
                d2 = line2(y2)
                if (d1 < x1 < d2) or (d1 < x2 < d2) or (d1 < (x1 + x2) / 2 < d2):
                    self.drawtext(src, (x1, y1), (x2, y2), self.classlist[class_id] + ' ' + str(round(distance, 2)),
                                  index=class_id, color=(0, 0, 255))
                else:
                    self.drawtext(src, (x1, y1), (x2, y2), self.classlist[class_id] + ' ' + str(round(distance, 2)),
                                  index=class_id, color=self.colors[class_id])
            else:
                self.drawtext(src, (x1, y1), (x2, y2), self.classlist[class_id] + ' ' + str(round(distance, 2)),
                              index=class_id, color=self.colors[class_id])

            dst = cv2.resize(src, (round(width / prop), round(height / prop)))

        if ifshow:
            cv2.imshow('result', dst)
            cv2.waitKey(0)
        return dst, iffall

    def img_identify2(self, img, ifshow=True, model_pt="yolov5s.pt", iou=0.45, conf=0.25,
                      isevening=False , speed = 0) -> np.ndarray:
        '''	@func: 图片识别
            @para	img: 图片路径或者图片数组
                    ifshow: 是否显示图片
            @return: 图片数组
        '''
        if type(img) == str:
            src = cv2.imread(img)
        else:
            src = img
        if isevening:  # 开启夜晚模式可以使用Gammmer函数均衡图像
            src = adjust_gamma(src, gamma=1.6)
        height, width, _ = src.shape  # 注意输出的尺寸是先高后宽
        _max = max(width, height)
        resized = np.zeros((_max, _max, 3), np.uint8)
        resized[0:height, 0:width] = src  # 将图片转换成正方形，防止后续图片预处理(缩放)失真
        # 图像预处理函数,缩放裁剪,交换通道  img     scale              out_size              swapRB
        blob = cv2.dnn.blobFromImage(resized, 1 / 255.0, (self.img_size, self.img_size), swapRB=True)
        prop = _max / self.img_size  # 计算缩放比例
        dst = cv2.resize(src, (round(width / prop), round(height / prop)))
        # print(prop)  # 注意，这里不能取整，而是需要取小数，否则后面绘制框的时候会出现偏差
        self.net.setInput(blob)  # 将图片输入到模型
        out = self.net.forward()  # 模型输出
        out = np.array(out[0])
        out = out[out[:, 4] >= conf]  # 利用numpy的花式索引,速度更快, 过滤置信度低的目标
        boxes = out[:, :4]
        confidences = out[:, 4]
        class_ids = np.argmax(out[:, 5:], axis=1)
        class_scores = np.max(out[:, 5:], axis=1)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf, iou)  # 非极大值抑制, 获取的是索引
        print(('检测到{}个物体'.format(len(indexes))))
        iffall = True if len(indexes) != 0 else False
        distance_warn = f(speed) # 对速度进行危险距离的判断
        list = []
        num = 0
        warn_list = []
        # 重写 cv2.举行矩阵
        src = np.ascontiguousarray(src)
        # self.drawtext(src, (x1, y1), (x2, y2), self.classlist[class_id] + ' ' + str(dist), index=class_id)
        self.drawLine(src, 0)  # 画危险区域

        for (i, c) in zip(indexes, confidences):  # 遍历每一个目标, 绘制目标框 + 距离
            box = boxes[i]
            class_id = class_ids[i]
            score = round(class_scores[i], 2)
            x1 = round((box[0] - 0.5 * box[2]) * prop)
            y1 = round((box[1] - 0.5 * box[3]) * prop)
            x2 = round((box[0] + 0.5 * box[2]) * prop)
            y2 = round((box[1] + 0.5 * box[3]) * prop)

            distance = round(self.mydistance(src, [x1, y1, x2, y2]), 2)

            # print(self.classlist[class_id], x1, y1, x2, y2, score, dist)

            list.append(self.classlist[class_id] + ' ' + str(round((x1 + x2) / 2)) + ' ' + str(
                round((y1 + y2) / 2)) + ' '+str(score)+ ' ' + str(distance)+'米')
            # self.drawtext(src, (x1, y1), (x2, y2), self.classlist[class_id] + ' ' + str(distance), index=class_id,
            #               color=self.colors[class_id])
            if distance < distance_warn and y2 > 400:  # 当距离过近且距离较低时
                d1 = line1(y2)
                d2 = line2(y2)
                # if (d1 < x1 < d2) or (d1 < x2 < d2) or (d1 < (x1 + x2) / 2 < d2):
                if d1 < (x1 + x2) / 2 < d2:
                    self.drawtext(src, (x1, y1), (x2, y2), self.classlist[class_id] + ' ' + str(distance),
                                  index=class_id, color=(0, 0, 255))
                    if distance != 0:
                        num += 1
                        warn_list.append([self.classlist_han[class_id], self.liangci[class_id], distance])
                else:
                    self.drawtext(src, (x1, y1), (x2, y2), self.classlist[class_id] + ' ' + str(distance),
                                  index=class_id, color=self.colors[class_id])
            else:
                self.drawtext(src, (x1, y1), (x2, y2), self.classlist[class_id] + ' ' + str(distance),
                              index=class_id, color=self.colors[class_id])
            dst = cv2.resize(src, (round(width / prop), round(height / prop)))
        if ifshow:
            cv2.imshow('result', dst)
            cv2.waitKey(0)
        return dst, iffall, list, num , warn_list

    def video_identify(self, video_path: str) -> None:
        '''	@func: 视频识别
            @para  video_path: 视频路径
            @return: None
        '''
        cap = cv2.VideoCapture(video_path)
        # print(type(cv2.CAP_PROP_FPS))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("此视频帧率{}".format(fps))
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            # 键盘输入空格暂停，输入q退出
            key = cv2.waitKey(1) & 0xff
            if key == ord(" "): cv2.waitKey(0)
            if key == ord("q"): break
            if not ret: break
            img, res = self.img_identify(frame, False , 100)
            cv2.imshow('result', img)

            if (time.time() - start_time) != 0:  # 实时显示帧数
                print(["FPS: ", 1 / (time.time() - start_time)])
                start_time = time.time()
            # time.sleep(1 / fps)  # 按原帧率播放
            if cv2.waitKey(int(1000 / fps)) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def drawtext(image, pt1, pt2, text, index, color=(255, 255, 255)):
        '''	@func: 根据给出的坐标和文本,在图片上进行绘制
            @para	image: 图片数组; pt1: 左上角坐标; pt2: 右下角坐标; text: 矩形框上显示的文本,即类别信息
            @return: None
        '''
        fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL  # 字体
        # fontFace = cv2.FONT_HERSHEY_COMPLEX  # 字体
        fontScale = 2  # 字体大小
        line_thickness = 3  # 线条粗细
        font_thickness = 3  # 文字笔画粗细

        # 绘制矩形框
        cv2.rectangle(image, pt1, pt2, color=color, thickness=line_thickness)
        # 计算文本的宽高: retval:文本的宽高; baseLine:基线与最低点之间的距离(本例未使用)
        retval, baseLine = cv2.getTextSize(text, fontFace=fontFace, fontScale=fontScale, thickness=font_thickness)
        # 计算覆盖文本的矩形框坐标
        topleft = (pt1[0], pt1[1] - retval[1])  # 基线与目标框上边缘重合(不考虑基线以下的部分)
        bottomright = (topleft[0] + retval[0], topleft[1] + retval[1])
        cv2.rectangle(image, topleft, bottomright, thickness=-1, color=(255, 255, 255))  # 绘制矩形框(填充)
        # 绘制文本
        cv2.putText(image, text, pt1, fontScale=fontScale, fontFace=fontFace, color=color, thickness=font_thickness)

    @staticmethod
    def drawLine(image, speed):
        line_thickness = 2  # 线条粗细
        font_thickness = 2  # 文字笔画粗细
        # 图像大小为640*384 ,206,235,135
        cv2.line(image, (300, 1080), (900, 400), color=(206,235,135), thickness=line_thickness)
        cv2.line(image, (1620, 1080), (1020, 400), color=(206,235,135), thickness=line_thickness)
        cv2.line(image, (900, 400), (1020, 400), color=(206,235,135), thickness=line_thickness)

    @staticmethod
    def drawtext2(image, pt1, pt2, text, index):
        '''	@func: 根据给出的坐标和文本,在图片上进行绘制
            @para	image: 图片数组; pt1: 左上角坐标; pt2: 右下角坐标; text: 矩形框上显示的文本,即类别信息
            @return: None
        '''
        fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL  # 字体
        # fontFace = cv2.FONT_HERSHEY_COMPLEX  # 字体
        fontScale = 2  # 字体大小
        line_thickness = 2  # 线条粗细
        font_thickness = 2  # 文字笔画粗细

        colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0)]
        # 绘制矩形框
        # cv2.line(image, pt1, pt2, colors[index])
        cv2.rectangle(image, pt1, pt2, color=colors[index], thickness=line_thickness)
        # 计算文本的宽高: retval:文本的宽高; baseLine:基线与最低点之间的距离(本例未使用)
        retval, baseLine = cv2.getTextSize(text, fontFace=fontFace, fontScale=fontScale, thickness=font_thickness)
        # 计算覆盖文本的矩形框坐标
        topleft = (pt1[0], pt1[1] - retval[1])  # 基线与目标框上边缘重合(不考虑基线以下的部分)
        bottomright = (topleft[0] + retval[0], topleft[1] + retval[1])
        cv2.rectangle(image, topleft, bottomright, thickness=-1, color=(0, 0, 255))  # 绘制矩形框(填充)
        # 绘制文本
        cv2.putText(image, text, pt1, fontScale=fontScale, fontFace=fontFace, color=colors[index],
                    thickness=font_thickness)

    def mydistance(self, image, box):  # boxs -> [x1,y1, x2, y2] 计算几步
        h, w, _ = image.shape  # 默认为 1920 * 1080
        px =  1
        dist = 100
        f = 1
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, y2)
        for i, p in enumerate(self.pre_pixel):
            if p <= center[1]:
                if i == 0:
                    dist = 0
                    break
                else:
                    dist_top = self.pre_dist[i]
                    dist_bottom = self.pre_dist[i - 1]
                    piexl_top = self.pre_pixel[i]
                    piexl_bottom = self.pre_pixel[i - 1]
                    piexl_sum = piexl_bottom - piexl_top
                    dist_sum = dist_top - dist_bottom
                    nomr = dist_sum / piexl_sum
                    dist = dist_top - nomr * (p - center[1])

                hang = abs(center[0] - w / 2) / abs(h - center[1]) * dist
                dist = (dist ** 2 + hang ** 2) ** 0.5* px  / f
                break
        return dist
