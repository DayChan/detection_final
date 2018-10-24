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
#server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(server_addr)
server.connect(client_addr)
def detect_one_img(yolo, frame):
    '''
    start = time.time()
    image = Image.fromarray(frame)
    print("fromArrayTime: ",time.time() - start)
    '''
    start = time.time()
    #frame = frame.convert('RGB')
    print("convertTime: ", time.time() - start)
    result_image = yolo.detect_image(frame) # save result.npy
    
    '''
    start = time.time()
    result = np.asarray(result_image)
    print("asArrayTime: ", time.time() - start)s
    '''
    #cv2.imshow(result)
    
    start = time.time()
    result_image.save(filename)
    print("writeTime: ", time.time() - start)
    
    start = time.time()
    udp_send(result_image)
    print("sendTime: ", time.time() - start)

def udp_send(image):
    json_image = {"image": image}
    json_send = pickle.dumps(json_image)
    server.sendall(json_send)