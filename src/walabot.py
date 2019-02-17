# -*- coding: utf-8 -*-
'''
    Created on Sun Nov 11 10:46 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Mon Nov 21 16:15 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

__all__ = ['walabot']

import matplotlib.pyplot as plt
import WalabotAPI
import numpy as np
import os
from sys import platform
from matplotlib import animation

class walabot():
    def __init__(self):
        '''
        Walabot initialization
        '''
        self.walabot = WalabotAPI
        self.walabot.Init()
        self.walabot.SetSettingsFolder()

    def set(self, minR, maxR, resR, minTheta, maxTheta, resTheta, minPhi, maxPhi, resPhi, threshold, mti=False):
        '''
        Args:
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
        self.minR = minR
        self.maxR = maxR
        self.resR = resR
        self.minTheta = minTheta
        self.maxTheta = maxTheta
        self.resTheta = resTheta
        self.minPhi = minPhi
        self.maxPhi = maxPhi
        self.resPhi = resPhi
        self.threshold = threshold
        self.mti = mti

    def start(self):
        # Walabot configuration
        self.walabot.ConnectAny()
        self.walabot.SetProfile(self.walabot.PROF_SENSOR)
        self.walabot.SetArenaR(self.minR, self.maxR, self.resR)
        self.walabot.SetArenaTheta(self.minTheta, self.maxTheta, self.resTheta)
        self.walabot.SetArenaPhi(self.minPhi, self.maxPhi, self.resPhi)
        self.walabot.SetThreshold(self.threshold)

        # Ignore static reflector
        if self.mti:
            self.walabot.SetDynamicImageFilter(self.walabot.FILTER_TYPE_MTI)

        self.walabot.Start()
        self.walabot.StartCalibration()

    def __delete__(self):
        self.walabot.Stop()
        self.walabot.Disconnect()

    def scan(self):
        '''
        Scan once, and return raw image signal. Scan can be applied only when walabot is succesfully
        initialized and started up. The output signal is a one-dimension scalar which stores the energies
        from voxels from space. To visit corresponding energy value, idx = (row_num * ((k * col_num) + j)) + i
        '''
        self.walabot.Trigger()
        signal = self.walabot.GetRawImage()

        return signal

    def dimensions(self):
        '''
        Return the dimensions of collected signals in the form of [row_num, col_num, channel_num]
        '''
        return list(map(int, [(self.maxTheta - self.minTheta)/self.resTheta - 1, (self.maxPhi - self.minPhi)/self.resPhi - 1, (self.maxR - self.minR)/self.resR - 1]))

if __name__ == '__main__':
    Walabot = walabot()