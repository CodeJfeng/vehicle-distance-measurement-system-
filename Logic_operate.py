import sys
import threading

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication

from mainFarmeWork import Ui_MainWindow
from util.onnxTOyolo import Onnx_clf


class My_UI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("车辆道路对象检测与分析系统")
        self.clf = Onnx_clf()


    def push_photo_botton(self):
        import tkinter as tk
        from tkinter.filedialog import askopenfilename
        tk.Tk().withdraw()  # 隐藏主窗口, 必须要用，否则会有一个小窗口
        source = askopenfilename(title="打开保存的图片")

        label_width = self.org_img.width()
        label_height = self.org_img.height()

        if source.endswith('.jpg') or source.endswith('.png') or source.endswith('.bmp'):
            img = cv2.imread(source)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # 对原图进行转化为QImage
            temp_img = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            pixmap_img = QPixmap.fromImage(temp_img).scaled(label_width, label_height)
            self.org_img.setPixmap(pixmap_img)

            res, out = self.clf.img_identify(img, False)

            temp_img2 = QImage(res, res.shape[1], res.shape[0], res.shape[1] * 3, QImage.Format_RGB888)
            # 将图片转换为QPixmap方便显示
            pixmap_img2 = QPixmap.fromImage(temp_img2).scaled(label_width, label_height)
            self.press_img.setPixmap(pixmap_img2)

    def push_video_button(self):
        import tkinter as tk
        from tkinter.filedialog import askopenfilename
        tk.Tk().withdraw()  # 隐藏主窗口, 必须要用，否则会有一个小窗口
        source = askopenfilename(title="打开保存视频")
        label_width = self.org_img.width()
        label_height = self.org_img.height()
        if source.endswith('.mp4') or source.endswith('.avi'):
            cap = cv2.VideoCapture(source)
            fps = cap.get(cv2.CAP_PROP_FPS)
            while cap.isOpened():
                ret, frame = cap.read() #  -> Boolen Image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                temp_img = QImage(frame, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
                pixmap_img = QPixmap.fromImage(temp_img).scaled(label_width, label_height)
                self.org_img.setPixmap(pixmap_img)

                res, isfall = self.clf.img_identify(frame, False)
                temp_img2 = QImage(res, res.shape[1], res.shape[0], res.shape[1] * 3, QImage.Format_RGB888)
                # 将图片转换为QPixmap方便显示
                pixmap_img2 = QPixmap.fromImage(temp_img2).scaled(label_width, label_height)
                self.press_img.setPixmap(pixmap_img2)

                if cv2.waitKey(int(1000/fps)) == ord('q')  :
                    break
            cap.release()


    def run(self):
        self.select_photo.clicked.connect(self.push_photo_botton)
        self.select_vedio.clicked.connect(self.push_video_button)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = My_UI()
    window.show()
    window.run()
    sys.exit(app.exec_())