# -*- coding: utf-8 -*-
'''
    Created on Sat Dec 15 20:46 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Sat Dec 15 20:46 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import time
import cv2
import os
import numpy as np
from src.model.darknet import darknet
from src.walabot import walabot
from src.utils import *
from src.detect import detector

class capture():
    def __init__(self, minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mti=False):
        model = darknet("cfg/yolov3.cfg", 80)
        model.load_weight("src/yolov3-1-1.weights")
        model.cuda()
        model.eval()
        self.detector = detector(model)
        self.walabot = walabot()
        self.walabot.set(minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mti)
        self.walabot.start()

    def detect(self, frame):
        pred = self.detector.detect(frame)

        for prediction_ in pred:
            if prediction_[6] == 0:
                x_coord = (prediction_[2] - prediction_[0])/2
                y_coord = (prediction_[3] - prediction_[1])/2
                print ("Detected target at %d, %d"%(x_coord, y_coord))
                return True


    def start_capture(self, save_dir):
        start = time.clock()
        optical_cap = cv2.VideoCapture(1)
        headers = np.ndarray(self.walabot.dimensions())

        while 1:
            ret, frame = optical_cap.read()
            try:
                status = self.detect(frame)
            except:
                continue

            if status == True:
                signals = []
                frames = []
                clocks_frame = []
                clocks_signals = []
                for i in range(1, 20):
                    ret, frame = optical_cap.read()
                    clocks_frame.append(time.clock())
                    clocks_signals.append(time.clock())
                    frames.append(frame)
                    signals.append(self.walabot.scan())

                for i in range(1, 20):
                    _time = str(clocks_frame[i])
                    path = os.join.path(save_dir, _time + ".jpg")
                    cv2.imwrite(path, frames[i])

                    _time = str(clocks_signals[i])
                    path = os.join.path(save_dir, _time)
                    file = open(path, "wb")
                    headers.tofile(file)
                    np.ndarray(signals[i]).tofile(file)
                    file.close()
                print("Collection complete")

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        optical_cap.release()

    def draw_bbox(self, prediction, img):
        '''
            Args:
                 prediction       : (list) list that record the prediction bounding-box
                 img              : (ndarray) original image
            Returns:
                 Image with bounding-box on it
        '''
        for prediction_ in prediction:
            coord1 = tuple(map(int, prediction_[:2]))
            coord2 = tuple(map(int, prediction_[2:4]))
            cv2.rectangle(img, coord1, coord2, (0, 255, 0), 2)

        return img

    def detect_test(self):
        # Capture the camera
        cap = cv2.VideoCapture(0)

        while 1:
            ret, frame = cap.read()
            try:
                # Making prediction
                prediction = self.detector.detect(frame)

                # Only person class proposals are needed
                pred_bbox = []
                for prediction_ in prediction:
                    if prediction_[6] == 0:
                        pred_bbox.append(prediction_[:4])

                # Drawing bounding-box
                self.draw_bbox(pred_bbox, frame)

                # Press 'q' to exit
                cv2.imshow("target", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            except:
                cv2.imshow("target", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        cap.release()
