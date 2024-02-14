# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:26:52 2024

@author: daphn
"""

import cv2
import dlib

image = cv2.imread("taylor.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

color = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX

detector = dlib.get_frontal_face_detector()

predictor68 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor81 = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")


faceLocs = detector(image)

for index, faceLoc in enumerate(faceLocs):
    landmarks = predictor81(gray, faceLoc)
    
    for i in range(0,81):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.putText(image, str(i), (x,y), font, .3, color, 1, cv2.LINE_AA)
        cv2.circle(image, (x,y), 3, color, -1)
        
        
cv2.imshow("Facial Landmark Points",image)    
cv2.waitKey()