import tensorflow as tf
import ssl
import numpy
ssl._create_default_https_context = ssl._create_unverified_context
from yolo3_predict import detect_one_img
from yolo import YOLO
import json
import numpy as np
import cv2
import time
from PIL import Image

yolo =  YOLO(
        model_path = 'model_data/weights/X-ylo-104.h5',
        anchors_path = 'model_data/yolo_anchors.txt',
        classes_path = 'model_data/classes.txt',
        gpu_num = 1,
        )
global graph
graph = tf.get_default_graph()

def detection():
    
    start = time.time()
    frame = Image.open('./camera.jpg')
    print("readTime is: ", time.time() - start)    # recv file tmpframe.npy

    start = time.time()
    with graph.as_default():
        detect_one_img(yolo, frame)
    print("detectAllTime is: ", time.time() - start)     # long time pre, have save result frame as send.jpg

if __name__ == '__main__':
    '''
    while(True):
        try:
            detection()
        except Exception as e:
            print("Wait for the camera pic: ",e)
    '''
    detection()
