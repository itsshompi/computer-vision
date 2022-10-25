# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:26:05 2022

@author: Stitch
"""
import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

capture_flag = False
capture_id = 1
last_time = time.time()
while cap.isOpened():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if ret == True and capture_flag == True:
        cv2.imwrite('./chessboard/calibration/id_'+str(capture_id)+'.png', img)
        print("Imagen capturada con id:::" +str(capture_id))
        capture_id += 1
        capture_flag = False
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        last_time = time.time()
        
    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
	    break
        
    if(last_time + 5 < time.time() and capture_flag == False):
        capture_flag = True
        print("Capture Flag is True")
        

cv2.destroyAllWindows()
cap.release()