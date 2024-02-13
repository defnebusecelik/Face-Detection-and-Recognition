# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:53:26 2024

@author: daphn"""


import cv2
import matplotlib.pyplot as plt

test_img=plt.imread("lana.jpg")

face_cascade=cv2.CascadeClassifier("frontalface.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")

test_img=cv2.cvtColor(test_img,cv2.COLOR_RGB2BGR)


#plt.figure(figsize=(12,8))
#plt.imshow(test_img)
#plt.show()


faces_test_img= face_cascade.detectMultiScale(test_img, 1.3, 3) # x,y,w,h
eye_test_img=eye_cascade.detectMultiScale(test_img,1.3,3)

for (x,y,w,h) in faces_test_img:
    cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,148,50), 3)
for (x,y,w,h) in eye_test_img:
    cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,148,50), 2)

cv2.imshow("lana del rey",test_img)
cv2.waitKey()

cv2.imwrite("eye.png",eye_test_img)