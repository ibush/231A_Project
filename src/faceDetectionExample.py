# From http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #TODO: Don't think it's finding these...
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

print face_cascade


img = cv2.imread('../video/still.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
for (x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()