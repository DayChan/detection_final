#!/usr/bin/python
#-*-coding:utf-8-*-

import zmq 
import cv2
import time
import pickle
context = zmq.Context()  
socket = context.socket(zmq.PUB)  
socket.bind("tcp://127.0.0.1:5000")
print("Port 5000 connect well")  
while True:  
    msg = cv2.imread('picture.jpg')
    msg = pickle.dumps(msg)
    socket.send(msg)
    print("Send successfully")
    time.sleep(0.2)