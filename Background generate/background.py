import numpy as np
import cv2 as cv
from math import exp,sqrt
import matplotlib.pyplot as plt
m2pix=512/2
ROBN=10
im_size_x=int(2*m2pix)
im_size_y=im_size_x
R=int(0.04*m2pix)

def gauss(x):
    a = 1.0 # amplititude of peak
    b = im_size_x/2.0 # center of peak
    c = im_size_x/130# standard deviation
    return a*exp(-((x-b)**2)/(2*(c**2)))
im=np.zeros((im_size_x,im_size_y))
data=[]
for i in range(0,R):
    cv.line(im,(im_size_x//2+i,0),(im_size_x//2+i,im_size_y),gauss(im_size_x/2-i),1)
    cv.line(im,(im_size_x//2-i,0),(im_size_x//2-i,im_size_y),gauss(im_size_x/2-i),1)

hl=int(0.94*512/2) #75 is small side of rectangle


cv.imshow('im',im)
cv.waitKey()


for i in range(0,im_size_x):
    for j in range(0,im_size_y):
        im[i,j]=(im[i,j])*255

cv.imwrite("ph.png",im)
