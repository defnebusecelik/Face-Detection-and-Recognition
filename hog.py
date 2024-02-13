# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:18:10 2024

@author: daphn
"""

import cv2
from skimage.feature import hog
from skimage import exposure


imagep="lana.jpg"
image = cv2.imread(imagep)

if len(image.shape) == 3:
    channel_axis = 2 
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



_ , hogImage=hog(grayscale_image,visualize=True)
rescaledImage=exposure.rescale_intensity(hogImage, in_range=(0,10))

cv2.imshow("HOG",hogImage)
cv2.imshow("Rescaled Image", rescaledImage)
cv2.waitKey()