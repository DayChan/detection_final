
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
import zmq
import threading

WIDTH=416
HEIGHT=416

REMOTE_IP = "192.168.31.46"
class OpencvWidget(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(OpencvWidget, self).__init__(*args, **kwargs)
        self.fps = 60
        self.createUI()
        self.classes_path = './classes.txt'
        self.class_names = self._get_class()
        self.generate_colors()
        self.pickle_data = 0                # pickle_data
        self.receive_content = 0            # 接收的content
        self.result = 0                     # result
        self.frame_cv2 = 0
        self.frame_PIL_colorL = 0
        self.frame_PIL = 0                  # frame_PIL
        self.frame_context = zmq.Context()
        self.frame_socket = self.frame_context.socket(zmq.SUB)
        self.result_context = zmq.Context()
        self.result_socket = self.result_context.socket(zmq.SUB)
        while True:
            try:
                self.frame_socket.connect("tcp://"+REMOTE_IP+":5000")
                print("Frame socket connect successfully! ")
                break
            except Exception as e:
                print("Wait for Camera Publisher.")
                print(e)
        self.frame_socket.setsockopt(zmq.SUBSCRIBE, b'')
        while True:
            try:
                self.result_socket.connect("tcp://"+REMOTE_IP+":5001")
                print("Result socket connect successfully! ")
                break
            except Exception as e:
                print("Wait for Result Publisher.")
                print(e)
        self.result_socket.setsockopt(zmq.SUBSCRIBE, b'')

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
            self.timer = QTimer(self, timeout=self.onCapture)
            self.timer.start(1000 / self.fps)       # 按帧数计算间隔时间
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def closeEvent(self, event):
        if hasattr(self, "timer"):
            self.timer.stop()
            self.timer.deleteLater()
            del self.predictor, self.detector, self.cascade, self.cap
        super(OpencvWidget, self).closeEvent(event)
        self.deleteLater()

    def onCapture(self):
        while self.frame_PIL == 0:
            print('Wait for frame in onCapture function.')
            time.sleep(1)
        while self.result == 0:
            print('Wait for result in onCapture function.')
            time.sleep(1)
        start = time.time()
        out_boxes = self.result[b"out_boxes"]
        out_classes = self.result[b"out_classes"]
        out_scores = self.result[b"out_scores"]

        frame = self.drawpicture(self.frame_PIL, out_boxes, out_classes, out_scores)
        frame = np.asarray(frame)

        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)

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

    def receive_frame_loop(self):
        while True:
            try:
                self.receive_content = self.frame_socket.recv()
                print(type(self.receive_content))
                time.sleep(1)
                self.frame_cv2 = pickle.loads(self.receive_content, encoding='bytes')
                cv2.imwrite("./picutre.jpg",self.frame_cv2)
                self.frame_PIL_colorL = Image.fromarray(self.frame_cv2)
                self.frame_PIL = self.frame_PIL_colorL.convert('RGB')
                print("Received frame.")
            except Exception as e:
                print("Receive frame error.")
                print(e)
                time.sleep(1)

    def receive_result_loop(self):
        while True:
            try:
                self.pickle_data = self.result_socket.recv()
                self.result = pickle.loads(self.pickle_data, encoding='bytes')
                print("Received result.")
            except Exception as e:
                print("Receive frame waiting.")
                print(e)
                time.sleep(1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    windows = OpencvWidget()
    windows.show()

    multi_thread_receive_frame_loop = threading.Thread(target=windows.receive_frame_loop)
    multi_thread_receive_frame_loop.start()     # 多开线程循环接收frame
    multi_thread_receive_result_loop = threading.Thread(target=windows.receive_result_loop)
    multi_thread_receive_result_loop.start()    # 多开线程循环接收result

    QTimer.singleShot(100, windows.start)             # 0.1秒后启动
    sys.exit(app.exec_())
