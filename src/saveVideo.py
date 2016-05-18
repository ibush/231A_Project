#From http://stackoverflow.com/questions/10605163/opencv-videowriter-under-osx-producing-no-output

import numpy as np
import cv2
import time


import cv2
import time

filename = '../video/video.avi'
FPS = 30 #TODO: This just sets the output speed, but it's not capturing that fast...
NUM_FRAMES = 120#3600

cap = cv2.VideoCapture(0)

size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.cv.FOURCC('8', 'B', 'P', 'S')
out = cv2.VideoWriter(filename, fourcc, FPS, size, True)


start = time.time()
for i in xrange(0, NUM_FRAMES) :
#while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)    #TODO: I think this write takes too long...only getting about 12 FPS
        #cv2.imshow('frame', frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break;

    else:
        print 'Error...'
        break;

end = time.time()
seconds = end - start
print "Time taken : {0} seconds".format(seconds)
fps  = NUM_FRAMES / seconds;
print "Estimated frames per second : {0}".format(fps);
cap.release()
out.release()
cv2.destroyAllWindows()
