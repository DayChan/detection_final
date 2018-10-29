import cv2 
import zmq 
import numpy as np
#import time
sub_port =6666 
context = zmq.Context() 
#connect to socket we subscrib 
socket_sub = context.socket(zmq.SUB) 
socket_sub.connect("tcp://localhost:%d" %sub_port) 
socket_sub.setsockopt(zmq.SUBSCRIBE, b"") 
while True: 
    #time.sleep(0.1)
    contents = socket_sub.recv() 
    nparr = np.asarray(bytearray(contents), dtype="uint8") 
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) 
    print img_decode.shape
    cv2.imshow('camera',img_decode) 
    cv2.waitKey(1)
