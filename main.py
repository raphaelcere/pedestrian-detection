#! /usr/local/bin/python
# -*- coding: utf-8 -*-

import sys
from glob import glob
import itertools as it

import numpy as np
import cv2,time

# load images
img1=cv2.imread(str(sys.argv[1:][0]))
#img2=cv2.imread(str(sys.argv[1:][1]))
#img3=cv2.imread(str(sys.argv[1:][2]))
#img4=cv2.imread(str(sys.argv[1:][3]))

# substract
sub1=cv2.subtract(img1,img2)
#sub2=cv2.subtract(img2,img3)

# show - press ESC to continue
cv2.imshow('sub', sub1)
print "ESC to continue"
ch = 0xFF & cv2.waitKey()
cv2.destroyAllWindows()

# save substract results
cv2.imwrite('_tmp.JPG',sub1)

# convert to greyscale
gray=cv2.cvtColor(sub1,cv2.COLOR_BGR2GRAY)

# apply Gaussian filter
blur = cv2.GaussianBlur(gray,(3,3),0)

# find contours with treshold
_, contours, _= cv2.findContours(blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
c=max(contours,key=cv2.contourArea)
print(cv2.contourArea(c))
if cv2.contourArea>20000:
   print("Object detected !")

# test detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

found, w = hog.detectMultiScale(blur, winStride=(8,8), padding=(32,32), scale=1.05)

print '%d found' % (len(found))
