# -*- coding: utf-8 -*-
'''
    Copyright(c) 2018
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
import datetime
from src.model.darknet import darknet
from src.walabot import walabot
from src.utils import *
from src.detect import detector

class capture():
    def __init__(self, minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mti=False):
        '''
        @description: Capture initialization. Walabot configuration is also completed here. Capture deploys a YOLO v3 detector to
        detect human and once a person is detected and its central point is near the centre of optical image, sensors
        will begin to collect data and save to predetermined directory.
        @arges:
             minR        : (int) scan arena configuration parameter, minimum distance
             maxR        : (int) maximum distance of scan arena
             resR        : (float) resolution of depth
             minTheta    : (int) minimum theta
             maxTheta    : (int) maximum theta
             resTheta    : (int) vertical angular resolution
             minPhi      : (int) minimum phi
             maxPhi      : (int) maximum phi
             resPhi      : (int) horizontal angular resolution
             threshold   : (int) threshold for weak signals
             mode        : (string) scan mode
             mti         : (boolean) ignore static reflectors
        '''
        # YOLO v3 detector deployment
        model = darknet("cfg/yolov3.cfg", 80)
        model.load_weight("yolov3.weights")
        model.cuda()
        model.eval()
        self.detector = detector(model)
        # Walabot configuration and start up
        #self.walabot = walabot()
        #self.walabot.set(minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mti)
        #self.walabot.start()

    def detect(self, frame, mid_point, offset):
        '''
        @description: Detector checks the central point of bounding-box. If it falls in the expected arena, sensors can start to
        collect data.
        @args:
            frame       : (ndarray) optical image frame
            mid_point   : (int) defines the central point of image
            offset      : (double) defines the scale of acceptable area
        '''
        pred = self.detector.detect(frame)

        for prediction_ in pred:
            if prediction_[6] == 0:                                                         # 0 denotes person class
                x_coord = (prediction_[2] - prediction_[0])/2
                y_coord = (prediction_[3] - prediction_[1])/2
                print ("Detected target at %d, %d"%(x_coord, y_coord))
                if abs(mid_point - x_coord) < offset:
                    return True

    def union_capture(self, save_dir, _lamda, frames_num):
        '''
        @description: union_capture interacts with C++ WALABOT capture programme using a text file. Detailed description
        about its parameters can infers to start_capture.
        @args:
            save_dir      : (string) save directory
            _lamda        : (double) defines the scale of acceptable area
            frames_num    : (int) expected acquisition frame number
        '''
        optical_cap = cv2.VideoCapture(1)                                                   # Capture on optical camera

        # Compute acceptable area
        ret, frame = optical_cap.read()
        mid_point = frame.shape[1] / 2
        offset = mid_point * _lamda
        file_dir = os.path.join(save_dir, "inter.txt")
        with open(file_dir, 'w') as file:
            file.write("0")
            file.close()

        while 1:
            ret, frame = optical_cap.read();
            try:
                status = self.detect(frame, mid_point, offset)                              # Human detection
            except:
                continue

            if status == True:                                                              # Send and receive signal
                with open(file_dir, 'w') as file:
                    file.write("1")
                    file.close()

                frames, clocks = [], []
                for i in range(0, 200):                                                     # Collect RGB image
                    ret, frame = optical_cap.read()
                    frames.append(frame)
                    _instant = "%d%d%d%d"%(datetime.datetime.now().hour, datetime.datetime.now().minute,
                                            datetime.datetime.now().second, datetime.datetime.now().microsecond)
                    _instant = _instant[:-3]
                                                                                            # Get instant time
                    clocks.append(time.clock())

                for i in range(0, 200):                                                     # Save images
                    _time = str(clocks[i])
                    path = os.path.join(save_dir, _time + ".jpg")
                    cv2.imwrite(path, frames[i])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        optical_cap.release()

    def start_capture(self, save_dir, _lamda, frames_num):
        '''
        @description: start_capture would call a optical camera and a walabot as sensors. It acquire optical images through
        the web camera successively and perform human detection based on optical image. If the bounding-box falls in the
        expected area, sensors begin to collect data and save them to predetermined directory. The filename of data is
        dependent on the CPU operation time. The acceptable area and expected acquisition frames can be customized.
        @args:
            save_dir      : (string) save directory
            _lamda        : (double) defines the scale of acceptable area
            frames_num    : (int) expected acquisition frame number
        '''
        optical_cap = cv2.VideoCapture(1)                                                   # Capture on optical camera

        headers = np.array(self.walabot.dimensions())                                       # Radio signal file headers

        # Compute acceptable area
        ret, frame = optical_cap.read()
        mid_point = frame.shape[1]/2
        offset = mid_point*_lamda

        while 1:
            ret, frame = optical_cap.read()
            try:
                status = self.detect(frame, mid_point, offset)                              # Human detection
            except:
                continue

            if status == True:
                signals = []                                                                # Signal list
                frames = []                                                                 # Optical image list
                clocks_frame = []
                clocks_signals = []
                # Data collection process
                for i in range(1, frames_num):
                    ret, frame = optical_cap.read()
                    clocks_frame.append(time.clock())
                    frames.append(frame)
                    #signals.append(self.walabot.scan()[0])
                    clocks_signals.append(time.clock())

                # Save collected data
                for i in range(1, frames_num):
                    # Save optical images
                    _time = str(clocks_frame[i - 1])
                    path = os.path.join(save_dir, _time + ".jpg")
                    cv2.imwrite(path, frames[i])

                    # Save radio signals by binary file
                    _time = str(clocks_signals[i])
                    path = os.path.join(save_dir, _time)                                    # Concatenate saving directory
                    #file = open(path, "wb")
                    #headers.tofile(file)
                    #np.array(signals[i]).tofile(file)
                    #file.close()
                print("Collection complete")

            if cv2.waitKey(1) & 0xFF == ord('q'):
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
        cap = cv2.VideoCapture(1)
        cap.set(3, 1080)
        cap.set(4, 1080)

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
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except:
                cv2.imshow("target", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()

if __name__ == "__main__":
    Capture = capture(10, 600, 10, -60, 60, 10, -60, 60, 10, 15)
    Capture.union_capture("F:/capturedata/", 0.7, 30)