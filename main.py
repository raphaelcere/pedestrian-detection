#! /usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2,time

# load images
img=cv2.imread("STC_0133.JPG")
img1=cv2.imread("STC_0134.JPG")

# substract
sub=cv2.subtract(img,img1)

# show - press ESC to continue
cv2.imshow('sub', sub)
ch = 0xFF & cv2.waitKey()
cv2.destroyAllWindows()

# save substract results
cv2.imwrite('_tmp.JPG',sub)

# convert to greyscale
gray=cv2.cvtColor(sub,cv2.COLOR_BGR2GRAY)

# apply Gaussian filter
blur = cv2.GaussianBlur(gray,(3,3),0)

# find contours
_, contours, _= cv2.findContours(blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
c=max(contours,key=cv2.contourArea)
print(cv2.contourArea(c))

if cv2.contourArea>20000:
   print("Object detected !")


# test detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)

print '%d found' % (len(found))
