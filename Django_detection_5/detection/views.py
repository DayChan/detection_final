from django.shortcuts import render
from django.http import HttpResponse
from django.http import FileResponse
import tensorflow as tf
import ssl
import numpy
ssl._create_default_https_context = ssl._create_unverified_context
from yolo3_predict import detect_one_img
from yolo import YOLO
from detection import detection
import json
import numpy as np
import cv2
import time

yolo =  YOLO(
        model_path = 'model_data/weights/X-ylo-104.h5',
        anchors_path = 'model_data/yolo_anchors.txt',
        classes_path = 'model_data/classes.txt',
        gpu_num = 1,
        )
global graph
graph = tf.get_default_graph()

def detection(request):
    
    start = time.time()
    file = request.FILES['file']
    frame = np.load(file)
    print("parseTimeRecv is: ", time.time() - start)    # recv file tmpframe.npy

    start = time.time()
    with graph.as_default():
        detect_one_img(yolo, frame)
    print("parseTimePre is: ", time.time() - start)     # long time pre, have save result frame as result.npy

    files = open('result.npy', 'rb')
    response =HttpResponse(files)
    response['Content-Type']='application/octet-stream' # put result.npy into response

    return response
