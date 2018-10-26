#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import time
import colorsys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2  # @UnresolvedImport 
import numpy as np
import socket
import json
import requests
import base64
from PIL import Image, ImageFont, ImageDraw
import pickle

WIDTH=416
HEIGHT=416
# BUF_SIZE = 102400

# client_addr = ('127.0.0.1', 9999)
# #client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.bind(client_addr)
# client.listen(1)
# sock, addr = client.accept()
URL = 'http://127.0.0.1:8000/detection/'
class OpencvWidget(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(OpencvWidget, self).__init__(*args, **kwargs)
        self.fps = 30
        self.createUI()
        self.classes_path = './classes.txt'
        self.class_names = self._get_class()
        self.generate_colors()
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
        # client.close()
        self.deleteLater()

    def onCapture(self):
        start= time.time()
        resp = requests.get(URL)
        rawdata = resp.content
        json_data = pickle.loads(rawdata)
        framePIL = json_data["framePIL"]
        out_boxes = json_data["out_boxes"]
        out_classes = json_data["out_classes"]
        out_scores = json_data["out_scores"]
        framePIL = framePIL.convert('RGB')
        frame = self.drawpicture(framePIL, out_boxes, out_classes, out_scores)
        frame = np.asarray(frame)
        '''
        self.strtoimage(data, 'test.jpg')
        frame = cv2.imread("test.jpg")
        '''
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        del frame
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        del frame
        '''
        self.videoView.setPixmap(QPixmap.fromImage(img))
        print("Time: ", time.time()-start)

    def strtoimage(self, str, filename):
        image_str = str.decode('ascii')
        image_byte = base64.b64decode(image_str)
        image_json = open('./'+filename, 'wb')
        image_json.write(image_byte)
        image_json.close()

    def drawpicture(self, image, out_boxes, out_classes, out_scores):
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate_colors(self):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        return 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OpencvWidget()
    w.show()
    # 0.1秒后启动
    QTimer.singleShot(100, w.start)
    sys.exit(app.exec_())
