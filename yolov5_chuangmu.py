import queue
import threading
import time

import cv2

from util.distance.Horizontal import Correction
from util.distance.mySGBM import mySGBM
from util.distance.setcameraconfig import cameraConfig
from util.onnxTOyolo import Onnx_clf


def distance(q, img_left, img_right, s):
    cor = Correction()
    height, width = img_left.shape[:2]
    map1x, map1y, map2x, map2y, Q = cor.getRectifyTransform(height,width,s)
    rectifyed_img1, rectifyed_img2 = cor.rectifyImage(img_left, img_right, map1x, map1y, map2x, map2y)
    my = mySGBM()
    disp, _  = my.stereoMatchSGBM(rectifyed_img1, rectifyed_img2, True)
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    q.put(points_3d)


if __name__=='__main__':
    clf = Onnx_clf()
    s = cameraConfig()
    s.setMiddleComfig() # 如果标定删
    q = queue.Queue()

    image_left = cv2.imread('data/distance/left/2.jpg')
    image_right = cv2.imread('data/distance/right/2.jpg')

    start = time.time()
    t = threading.Thread(target=distance, args=(q, image_left, image_right, s))
    t.start()

    img = image_left
    res, out = clf.img_identify2(img, t, q, False)
    print(time.time()-start)
    cv2.imshow("test",res)
    cv2.waitKey(0)



