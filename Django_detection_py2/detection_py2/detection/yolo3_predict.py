# -*- coding: utf-8 -*-
import sys
import time
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import socket
import json
import pickle
import time

filename = './send.jpg'
client_addr = ('127.0.0.1', 9999)
server_addr = ('127.0.0.1', 8888)
BUFSIZE = 65535
#server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#server.bind(server_addr)
#server.connect(client_addr)
def detect_one_img(yolo, frame):
    '''
    start = time.time()
    image = Image.fromarray(frame)
    print("fromArrayTime: ",time.time() - start)
    '''
    '''
    start = time.time()
    frame = frame.convert('RGB')
    print("convertTime: ", time.time() - start)
    '''
    out_boxes, out_scores, out_classes = yolo.detect_image(frame) # save result.npy
    return out_boxes, out_scores, out_classes
    '''
    start = time.time()
    result = np.asarray(result_image)
    print("asArrayTime: ", time.time() - start)s
    '''
    #cv2.imshow(result)

    '''
    start = time.time()
    result_image.save(filename)
    print("writeTime: ", time.time() - start)
    '''
    '''
    start = time.time()
    udp_send(result_image)
    print("sendTime: ", time.time() - start)
'''
'''
def udp_send(image):
    json_image = {"image": image}
    json_send = pickle.dumps(json_image,'dumpsfile.pickle')
    with open('./send.jpg', 'r') as infile:
        d = infile.read(BUFSIZE)
        client.sendto('start', client_addr)
        while(d):
            client.sendto(d, client_addr)
            d = infile.read(BUFSIZE)
        client.sendto('end', client_addr)
        '''
