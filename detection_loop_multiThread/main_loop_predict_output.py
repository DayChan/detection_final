# -*- coding: utf-8 -*-
import tensorflow as tf
import ssl
import numpy
ssl._create_default_https_context = ssl._create_unverified_context
from yolo import YOLO
import json
import numpy as np
import cv2
import zmq
import time
import pickle
import threading
import pickle
from PIL import Image
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

class DetectionLoop():
    def __init__(self):
        self.yolo = YOLO(
            model_path='model_data/weights/re_res_not_maxpooling_yolo7/trained_weights_stage200.h5',
            anchors_path='model_data/re_anchors.txt',
            classes_path='model_data/classes.txt',
            gpu_num=1,
        )
        self.graph = tf.get_default_graph()
        self.receive_content = 0        # 接收的content
        self.frame_cv2 = 0
        self.frame_PIL = 0
        self.receive_context = zmq.Context()
        self.receive_socket = self.receive_context.socket(zmq.SUB)
        self.send_context = zmq.Context()
        self.send_socket = self.send_context.socket(zmq.PUB)
        self.send_data = 0              # 初始化send_data
        self.multi_thread_send = 0      # 初始化multi_thread_send

        while True:
            try:
                self.receive_socket.connect("tcp://127.0.0.1:6666")
                print("Recieve socket connect successfully! ")
                break
            except Exception as e:
                print("Wait for Camera Publisher.")
                print(e)
        self.receive_socket.setsockopt(zmq.SUBSCRIBE, '')
        while True:
            try:
                self.send_socket.bind("tcp://0.0.0.0:5001")
                print("Send socket bind successfully! ")
                break
            except Exception as e:
                print("Wait for Port 5001 release.")
                print(e)

    def detection(self):
        while self.frame_PIL == 0:
            print('Wait for frame in detection function.')
            time.sleep(1)

        #start = time.time()
        with self.graph.as_default():
            out_boxes, out_scores, out_classes = self.yolo.detect_image(self.frame_PIL)
            if out_boxes.all():
	        print("out_boxes: ",out_boxes)
            if out_scores.all():
	        print("out_scores: ",out_scores)
            if out_classes.all():
	        print("out_classes: ",out_classes)
        #print("detectAllTime is: ", time.time() - start)
    def receive_frame_loop(self):
        while True:
            try:
                self.receive_content = self.receive_socket.recv()
		#print("Received data...")
                #self.frame_cv2 = pickle.loads(self.receive_content)
		nparr = np.asarray(bytearray(self.receive_content), dtype="uint8") 
    		self.frame_cv2 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) 
                self.frame_PIL = Image.fromarray(self.frame_cv2)
            except Exception as e:
                print("Receive frame error.")
                print(e)
                time.sleep(1)

    def sender(self, data):
        pickle_data = pickle.dumps(data)
        print(type(pickle_data))
        self.send_socket.send(pickle_data)
        #print("Send Fine.")


if __name__ == '__main__':
    detection_loop = DetectionLoop()
    multi_thread_receive_frame_loop = threading.Thread(target=detection_loop.receive_frame_loop)
    # multi_thread_receive_frame_loop.setDaemon(True)
    multi_thread_receive_frame_loop.start()     # 多开线程循环接收
    while True:
        try:
            # print("Going to step detection.")
            detection_loop.detection()
            # print("Detection Fine.")
        except Exception as e:
            print(e)
            time.sleep(1)
