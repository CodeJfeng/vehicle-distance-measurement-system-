# -*- coding: utf-8 -*-
import sys

import cv2
import numpy as np
# Form implementation generated from reading ui file 'mainFarmeWork.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from util.onnxTOyolo import Onnx_clf


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 1070, 650))
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setFlat(False)
        self.groupBox.setObjectName("groupBox")
        self.select_vedio = QtWidgets.QPushButton(self.groupBox)
        self.select_vedio.setGeometry(QtCore.QRect(310, 460, 151, 41))
        self.select_vedio.setObjectName("select_vedio")
        self.select_photo = QtWidgets.QPushButton(self.groupBox)
        self.select_photo.setGeometry(QtCore.QRect(150, 460, 121, 41))
        self.select_photo.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.select_photo.setMouseTracking(False)
        self.select_photo.setTabletTracking(False)
        self.select_photo.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.select_photo.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.select_photo.setObjectName("select_photo")
        self.evening = QtWidgets.QPushButton(self.groupBox)
        self.evening.setGeometry(QtCore.QRect(650, 460, 131, 41))
        self.evening.setObjectName("evening")
        self.select_groap = QtWidgets.QPushButton(self.groupBox)
        self.select_groap.setGeometry(QtCore.QRect(500, 460, 121, 41))
        self.select_groap.setObjectName("select_groap")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(850, 470, 71, 20))
        self.radioButton.setObjectName("radioButton")
        self.org_img = QtWidgets.QLabel(self.groupBox)
        self.org_img.setGeometry(QtCore.QRect(20, 40, 500, 350))
        self.org_img.setFrameShape(QtWidgets.QFrame.Box)
        self.org_img.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.org_img.setText("")
        self.org_img.setAlignment(QtCore.Qt.AlignCenter)
        self.org_img.setIndent(-2)
        self.org_img.setOpenExternalLinks(False)
        self.org_img.setObjectName("org_img")
        self.press_img = QtWidgets.QLabel(self.groupBox)
        self.press_img.setGeometry(QtCore.QRect(550, 40, 500, 350))
        self.press_img.setFrameShape(QtWidgets.QFrame.Box)
        self.press_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.press_img.setLineWidth(1)
        self.press_img.setText("")
        self.press_img.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.press_img.setObjectName("press_img")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1100, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "车辆道路对象检测与分析系统"))
        self.groupBox.setTitle(_translate("MainWindow", "目标检测"))
        self.select_vedio.setText(_translate("MainWindow", "上传视频"))
        self.select_photo.setText(_translate("MainWindow", "上传图像"))
        self.evening.setText(_translate("MainWindow", "夜间模式"))
        self.select_groap.setText(_translate("MainWindow", "打开摄像头"))
        self.radioButton.setText(_translate("MainWindow", "夜间模式"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec())