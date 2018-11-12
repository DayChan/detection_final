# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import *
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

''' line 76 model '''
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolov3_6600.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/classes.txt',
        "score" : 0.1,
        "iou" : 0.1,
        "model_image_size" : (608, 608),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)

        except Exception as e:
            print(e)
            self.yolo_model = res_not_pooling_tiny_yolo_body7(Input(shape=(None,None,1)), num_anchors//2, num_classes) \
                if is_tiny_version else not_pooling_tiny_yolo_body2(Input(shape=(None,None,1)), num_anchors//3, num_classes)#改了？
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

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

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        print("boxes:",boxes,"scores",scores,"classes:",classes)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data = image_data.reshape(image_data.shape[0],image_data.shape[0],1)#3->1,so reshape.#####
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        startPre = timer()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        filter_boxes(out_boxes, out_scores, out_classes)
        print("predictTime: ", timer() - startPre)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        '''
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

        end = timer()
        print("predictAllTime: ",end - start)
        '''
        return out_boxes, out_scores, out_classes

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def filter_boxes(out_boxes, out_scores, out_classes):
    """
    out_boxes = [[187.9777, -9.5165205, 777.2686, 902.68054],
                 [151.95294, 535.67737, 242.61598, 645.80597],
                 [197.97777, -9.5165205, 777.2686, 902.68054],
                 [104.001785, 664.8004, 191.59044, 790.4712]]
    out_scores = [0.30529305, 0.27804533, 0.15484284, 0.12531385]
    out_classes = [2, 4, 4, 4]
    out_boxes = np.array(out_boxes, dtype="float32")
    out_scores = np.array(out_scores, dtype="float32")
    out_classes = np.array(out_classes, dtype="int32")

    # TEST DATA ABOVE
    """
    mask = [0] * out_boxes.shape[0]
    delete_index = []
    for i in range(out_boxes.shape[0]):
        for j in range(i + 1, out_boxes.shape[0]):
            if mask[i] != 0:
                break
            elif mask[j] != 0:
                continue
            if iou(out_boxes[i], out_boxes[j]) > 0.6:
                if out_scores[i] < out_scores[j]:
                    mask[i] = 1
                else:
                    mask[j] = 1     # mask generate

    for i in range(len(mask)):
        if mask[i] != 0:
            delete_index.append(i)  # index list waiting for delete

    out_boxes = np.delete(out_boxes, delete_index, axis=0)
    out_scores = np.delete(out_scores, delete_index, axis=0)
    out_classes = np.delete(out_classes, delete_index, axis=0)  # 删除
    print(out_boxes, out_scores, out_classes)
    return out_boxes, out_scores, out_classes


def iou(box1, box2):
    cross_width = min(box1[3], box2[3]) - max(box1[1], box2[1])
    cross_height = min(box1[2], box2[2]) - max(box1[0], box2[0])
    if cross_width <= 0 or cross_height <= 0:
        return 0.0
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    cross_area = cross_height * cross_width
    return cross_area / (box1_area + box2_area - cross_area)
