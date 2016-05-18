import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA

WIDTH_FRACTION = 0.6 # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1
CASCADE_PATH = "haarcascade_frontalface_default.xml"
DEFAULT_VIDEO = "../video/iphone_2m.mov"
FPS = 30
WINDOW_TIME_SEC = 30
WINDOW_SIZE = WINDOW_TIME_SEC * FPS #2048
MIN_HR_BPM = 45.0
MAX_HR_BMP = 240.0
SEC_PER_MIN = 60

USE_SEGMENTATION = True
SEGMENTATION_HEIGHT_FRACTION = 1.2
SEGMENTATION_WIDTH_FRACTION = 0.8
GRABCUT_ITERATIONS = 5
    
def getROI(image, faceBox): 
    if USE_SEGMENTATION:
        widthFrac = SEGMENTATION_WIDTH_FRACTION
        heigtFrac = SEGMENTATION_HEIGHT_FRACTION
    else:
        widthFrac = WIDTH_FRACTION
        heigtFrac = HEIGHT_FRACTION

    # Adjust bounding box
    (x, y, w, h) = faceBox
    widthOffset = int((1 - widthFrac) * w / 2)
    heightOffset = int((1 - heigtFrac) * h / 2)
    faceBoxAdjusted = (x + widthOffset, y + heightOffset,
        int(widthFrac * w), int(heigtFrac * h))

    # Segment
    if USE_SEGMENTATION:
        mask = np.zeros(image.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        cv2.grabCut(image,mask,faceBoxAdjusted,bgdModel,fgdModel,GRABCUT_ITERATIONS,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        roi = image*mask2[:,:,np.newaxis]
        #plt.imshow(roi),plt.colorbar(),plt.show()
    else:
        (x, y, w, h) = faceBoxAdjusted

        # Show rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        roi = image[x:x+w, y:y+h, :]
        
    return roi

# Sum of square differences between x1, x2, y1, y2 points for each ROI
def distance(roi1, roi2):
    return sum((roi1[i] - roi2[i])**2 for i in range(len(roi1)))

def getBestROI(frame, faceCascade, previousFaceBox):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    roi = None
    faceBox = None

    # If no face detected, use ROI from previous frame
    if len(faces) == 0:
        faceBox = previousFaceBox

    # if many faces detected, use one closest to that from previous frame
    elif len(faces) > 1:
        if previousFaceBox is not None:
            # Find closest
            minDist = float("inf")
            for face in faces:
                if distance(previousFaceBox, face) < minDist:
                    faceBox = face
        else:
            # Chooses largest box by area (most likely to be true face)
            maxArea = 0
            for face in faces:
                if (face[2] * face[3]) > maxArea:
                    faceBox = face

    # If only one face dectected, use it!
    else:
        faceBox = faces[0]

    if faceBox is not None:
        roi = getROI(frame, faceBox)

    return faceBox, roi

def getHeartRate(window):
    # Normalize across the window to have zero-mean and unit variance
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std

    # Separate into three source signals using ICA
    ica = FastICA()
    srcSig = ica.fit_transform(normalized)

    # Find power spectrum
    powerSpec = np.abs(np.fft.fft(srcSig, axis=0))**2
    freqs = np.fft.fftfreq(WINDOW_SIZE, 1.0 / FPS)

    # Find heart rate
    maxPwrSrc = np.max(powerSpec, axis=1)
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
    validPwr = maxPwrSrc[validIdx]
    validFreqs = freqs[validIdx]
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx]
    print hr

    #return hr

    # Plot signals
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
    colors = ["r", "g", "b"]
    plt.figure()
    for i in range(3):
        plt.plot(seconds, normalized[:,i], colors[i])
    plt.xlabel("Time (sec)")
    plt.ylabel("Normalized color intensity")
    plt.show()

    plt.figure()
    for i in range(3):
        plt.plot(seconds, srcSig[:,i])
    plt.xlabel("Time (sec)")
    plt.ylabel("Source signal strength")
    plt.show()

    # Plot power spectrum
    idx = np.argsort(freqs)
    plt.figure()
    for i in range(3):
        plt.plot(freqs[idx], powerSpec[idx,i])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim([0.75, 4])
    plt.show()

'''
    # Find power spectrum of raw signals
    powerSpecRaw = np.abs(np.fft.fft(normalized, axis=0))**2

    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)
        #plt.xlim([0, 4])
        plt.plot(freqs[idx], powerSpecRaw[idx,i])
    plt.show()
'''

# Set up video and fact tracking
try:
    videoFile = sys.argv[1]
except:
    videoFile = DEFAULT_VIDEO  
video = cv2.VideoCapture(videoFile)
faceCascade = cv2.CascadeClassifier(CASCADE_PATH)

colorSig = [] # Will store the average RGB color values in each frame's ROI
heartRates = [] # Will store the heart rate calculated every 1 second
previousFaceBox = None
while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        break
    #print np.shape(frame) 720x1280x3

    previousFaceBox, roi = getBestROI(frame, faceCascade, previousFaceBox)

    if roi is not None: # Will be True unless first frame and no face detected
        avgColor = np.mean(roi, (0,1))
        colorSig.append(avgColor)

    # Calculate heart rate every one second (once have 30-second of data)
    if (len(colorSig) >= WINDOW_SIZE) and (len(colorSig) % FPS == 0):
        windowStart = len(colorSig) - WINDOW_SIZE
        window = colorSig[windowStart : windowStart + WINDOW_SIZE]
        heartRates.append(getHeartRate(window))

    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    previousROI = roi

print heartRates
video.release()
cv2.destroyAllWindows()