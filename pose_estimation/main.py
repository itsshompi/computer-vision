# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 18:23:21 2022

@author: Stitch
"""

import numpy as np
import cv2 as cv
import glob
# Load previously saved data

with np.load('calibration_ouput.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    
    
    