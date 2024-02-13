# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:10:30 2024

@author: daphn
"""

import cv2
import matplotlib.pyplot as plt

test_img=plt.imread("taylor.jpg")
test_img2=plt.imread("lana.jpg")

face_cascade=cv2.CascadeClassifier("frontalface.xml")

test_img=cv2.cvtColor(test_img,cv2.COLOR_RGB2BGR)
test_img2=cv2.cvtColor(test_img2,cv2.COLOR_RGB2BGR)

#plt.figure(figsize=(12,8))
#plt.imshow(test_img)
#plt.show()

gray_test_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
gray_test_img2=cv2.cvtColor(test_img2,cv2.COLOR_BGR2GRAY)


faces_test_img_1 = face_cascade.detectMultiScale(test_img, 1.1, 3) # x,y,w,h
faces_test_img_2 = face_cascade.detectMultiScale(test_img2, 1.3, 3) # x,y,w,h

for (x,y,w,h) in faces_test_img_1:
    cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,148,50), 3)


for (x1,y1,w1,h1) in faces_test_img_2:
    cv2.rectangle(test_img2, (x1,y1), (x1+w1,y1+h1), (255,148,50), 3)

cv2.imshow("taylor swift",test_img)
cv2.imshow("lana del rey",test_img2)
cv2.waitKey()