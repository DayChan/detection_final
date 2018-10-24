#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2  # @UnresolvedImport 
import numpy as np
import socket
import json
import base64
from PIL import Image
import pickle

WIDTH=416
HEIGHT=416
BUF_SIZE = 102400

client_addr = ('127.0.0.1', 9999)
#client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.bind(client_addr)
client.listen(1)
sock, addr = client.accept()
class OpencvWidget(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(OpencvWidget, self).__init__(*args, **kwargs)
        self.fps = 30
        self.createUI()
        
    def createUI(self):
        self.resize(800, 600)
        self.setWindowTitle("Classifier")
        self.videoView = QLabel("稍候，正在初始化数据和摄像头。。。")
        self.videoView.setAlignment(Qt.AlignCenter)
        self.text_result1 = QLabel("")
        self.text_result2 = QLabel("")
        self.text_result3 = QLabel("")
        self.text_result4 = QLabel("")
        self.text_result5 = QLabel("")
        self.text_result1.setAlignment(Qt.AlignCenter)
        self.text_result2.setAlignment(Qt.AlignCenter)
        self.text_result3.setAlignment(Qt.AlignCenter)
        self.text_result4.setAlignment(Qt.AlignCenter)
        self.text_result5.setAlignment(Qt.AlignCenter)
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.videoView)
        self.vlayout.addWidget(self.text_result1)
        self.vlayout.addWidget(self.text_result2)
        self.vlayout.addWidget(self.text_result3)
        self.vlayout.addWidget(self.text_result4)
        self.vlayout.addWidget(self.text_result5)
        self.widget = QWidget()
        self.widget.setLayout(self.vlayout)
        self.setCentralWidget(self.widget)

    def start(self):
        try:
            '''
            self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                return QMessageBox.critical(self, "错误", "打开摄像头失败")
            # 开启定时器定时捕获
            '''
            self.timer = QTimer(self, timeout = self.onCapture)
            self.timer.start(1000 / self.fps)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def closeEvent(self, event):
        if hasattr(self, "timer"):
            self.timer.stop()
            self.timer.deleteLater()
            self.cap.release()
            del self.predictor, self.detector, self.cascade, self.cap
        super(OpencvWidget, self).closeEvent(event)
        client.close()
        self.deleteLater()

    def onCapture(self):
        rawdata = sock.recv(BUF_SIZE)
        json_data = pickle.loads(rawdata)
        image_str = json_data["image"]
        frame = Image.fromstring(image_str)
        frame = np.asarray(frame)
        '''
        self.strtoimage(data, 'test.jpg')
        frame = cv2.imread("test.jpg")
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        del frame
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        del frame
        '''
        self.videoView.setPixmap(QPixmap.fromImage(img))

    def strtoimage(self, str, filename):
        image_str = str.decode('ascii')
        image_byte = base64.b64decode(image_str)
        image_json = open('./'+filename, 'wb')
        image_json.write(image_byte)
        image_json.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OpencvWidget()
    w.show()
    # 0.1秒后启动
    QTimer.singleShot(100, w.start)
    sys.exit(app.exec_())
