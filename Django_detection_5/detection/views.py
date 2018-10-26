from django.shortcuts import render
from django.http import HttpResponse
from django.http import FileResponse
import tensorflow as tf
import ssl
import numpy
ssl._create_default_https_context = ssl._create_unverified_context
from yolo3_predict import detect_one_img
from yolo import YOLO
from PIL import Image
import json
import numpy as np
import cv2
import time
import pickle

yolo = YOLO(
        model_path='model_data/weights/X-ylo-104.h5',
        anchors_path='model_data/yolo_anchors.txt',
        classes_path='model_data/classes.txt',
        gpu_num=1,
        )
global graph
graph = tf.get_default_graph()

def detection(request):
    start = time.time()
    framePIL = Image.open('./camera.jpg')
    print("readTime is: ", time.time() - start)  # recv file tmpframe.npy

    start = time.time()
    with graph.as_default():
        out_boxes, out_scores, out_classes = detect_one_img(yolo, framePIL)
    print("detectAllTime is: ", time.time() - start)  # long time pre, have save result frame as send.jpg

    send_data = {
        "framePIL": framePIL,
        "out_boxes": out_boxes,
        "out_scores": out_scores,
        "out_classes": out_classes
    }
    start = time.time()
    pickle_data = pickle.dumps(send_data)
    print("pickleTime: ", time.time() - start)
    response = HttpResponse(pickle_data)

    return response
