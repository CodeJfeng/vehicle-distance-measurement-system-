import sys
import threading

import cv2
import numpy as np
import pyttsx3
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication

from ui.QT.main import Ui_mainWindow
from util.onnxTOyolo import Onnx_clf

def play_sound(engine, name, liangci , dist):
    text = '前方{}米处有一{}，注意避让'.format(round(dist), liangci+name)
    engine.say(text)
    engine.runAndWait()



class My_UI(QMainWindow, Ui_mainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_mainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("车辆道路对象检测与分析系统")
        self.clf = Onnx_clf()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 250)
        self.t = threading.Thread(target=play_sound, args=(self.engine, '', '',''))


    def push_photo_botton(self):
        import tkinter as tk
        from tkinter.filedialog import askopenfilename
        tk.Tk().withdraw()  # 隐藏主窗口, 必须要用，否则会有一个小窗口
        source = askopenfilename(title="打开保存的图片")

        label_width = self.org_img.width()
        label_height = self.org_img.height()

        if source.endswith('.jpg') or source.endswith('.png') or source.endswith('.bmp'):
            img = cv2.imread(source)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # 对原图进行转化为QImage
            temp_img = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            pixmap_img = QPixmap.fromImage(temp_img).scaled(label_width, label_height)
            self.org_img.setPixmap(pixmap_img)

            res, out, list, num , warn_list = self.clf.img_identify2(img, False ,model_pt='yolov5s.pt', iou=self.iouSpinBox.value(),
                                              conf = self.confSpinBox.value(), isevening = self.select_evening.isChecked(),
                                              speed = self.speedSpinBox.value())

            if not num == 0 and not self.t.is_alive():
                warn_list.sort(key=lambda x: x[2])
                # print(warn_list)
                self.t = threading.Thread(target=play_sound,
                                          args=(self.engine, warn_list[0][0], warn_list[0][1], warn_list[0][2]))
                self.t.start()


            self.resultWidget.clear()
            self.resultWidget.addItems(list)

            res = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)


            temp_img2 = QImage(res, res.shape[1], res.shape[0], res.shape[1] * 3, QImage.Format_RGB888)
            # 将图片转换为QPixmap方便显示
            pixmap_img2 = QPixmap.fromImage(temp_img2).scaled(label_width, label_height)
            self.org_img.setPixmap(pixmap_img2)

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
                ret, img = cap.read()
                # 对原图进行转化为QImage
                temp_img = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
                pixmap_img = QPixmap.fromImage(temp_img).scaled(label_width, label_height)
                self.org_img.setPixmap(pixmap_img)

                res, out, list, num, warn_list = self.clf.img_identify2(img, False, model_pt='yolov5s.pt',
                                                                        iou=self.iouSpinBox.value(),
                                                                        conf=self.confSpinBox.value(),
                                                                        isevening=self.select_evening.isChecked(),
                                                                        speed=self.speedSpinBox.value())

                self.resultWidget.clear()
                self.resultWidget.addItems(list)

                res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

                temp_img2 = QImage(res, res.shape[1], res.shape[0], res.shape[1] * 3, QImage.Format_RGB888)
                # 将图片转换为QPixmap方便显示
                pixmap_img2 = QPixmap.fromImage(temp_img2).scaled(label_width, label_height)
                self.org_img.setPixmap(pixmap_img2)

                if not num == 0 and not self.t.is_alive():
                    warn_list.sort(key = lambda x : x[2])
                    # print(warn_list)
                    self.t = threading.Thread(target=play_sound, args=(self.engine, warn_list[0][0], warn_list[0][1],warn_list[0][2]))
                    self.t.start()

                if cv2.waitKey(int(1000/fps)) == ord('q') :
                    break
            cap.release()

    def push_camera_button(self):
        label_width = self.org_img.width()
        label_height = self.org_img.height()
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, img = cap.read()
            # 对原图进行转化为QImage
            temp_img = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            pixmap_img = QPixmap.fromImage(temp_img).scaled(label_width, label_height)
            self.org_img.setPixmap(pixmap_img)

            res, out, list, num, warn_list = self.clf.img_identify2(img, False, model_pt='yolov5s.pt',
                                                                    iou=self.iouSpinBox.value(),
                                                                    conf=self.confSpinBox.value(),
                                                                    isevening=self.select_evening.isChecked(),
                                                                    speed=self.speedSpinBox.value())

            self.resultWidget.clear()
            self.resultWidget.addItems(list)

            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

            temp_img2 = QImage(res, res.shape[1], res.shape[0], res.shape[1] * 3, QImage.Format_RGB888)
            # 将图片转换为QPixmap方便显示
            pixmap_img2 = QPixmap.fromImage(temp_img2).scaled(label_width, label_height)
            self.org_img.setPixmap(pixmap_img2)

            if not num == 0 and not self.t.is_alive():
                warn_list.sort(key=lambda x: x[2])
                # print(warn_list)
                self.t = threading.Thread(target=play_sound,
                                          args=(self.engine, warn_list[0][0], warn_list[0][1], warn_list[0][2]))
                self.t.start()

            if cv2.waitKey(int(1000 / fps)) == ord('q'):
                break
        cap.release()


    def iou_change(self):
        self.iouSpinBox.setValue(self.iouSlider.value()/100)

    def conf_change(self):
        self.confSpinBox.setValue(self.confSlider.value()/100)

    def speed_change(self):
        self.speedSpinBox.setValue(self.SpeedSlider.value())

    def run(self):
        self.select_picture.clicked.connect(self.push_photo_botton)
        self.select_video.clicked.connect(self.push_video_button)
        self.select_camera.clicked.connect(self.push_camera_button)
        self.iouSlider.valueChanged.connect(self.iou_change)
        self.confSlider.valueChanged.connect(self.conf_change)
        self.SpeedSlider.valueChanged.connect(self.speed_change)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = My_UI()
    window.show()
    window.run()
    sys.exit(app.exec_())