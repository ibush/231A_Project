import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA
import warnings

USE_SEGMENTATION = False
REMOVE_EYES = False
FOREHEAD_ONLY = False

CASCADE_PATH = "haarcascade_frontalface_default.xml"
VIDEO_DIR = "../video/"
DEFAULT_VIDEO = "iphone_2m.mov"
RESULTS_SAVE_DIR = "../results/" + ("segmentation/" if USE_SEGMENTATION else "no_segmentation/")
if REMOVE_EYES:
    RESULTS_SAVE_DIR += "no_eyes/"
if FOREHEAD_ONLY:
    RESULTS_SAVE_DIR += "forehead/"

MIN_FACE_SIZE = 100

WIDTH_FRACTION = 0.6 # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1

FPS = 14.99
WINDOW_TIME_SEC = 30
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))
MIN_HR_BPM = 45.0
MAX_HR_BMP = 240.0
MAX_HR_CHANGE = 12.0
SEC_PER_MIN = 60

SEGMENTATION_HEIGHT_FRACTION = 1.2
SEGMENTATION_WIDTH_FRACTION = 0.8
GRABCUT_ITERATIONS = 5

EYE_LOWER_FRAC = 0.25
EYE_UPPER_FRAC = 0.5

def segment(image, faceBox):
    mask = np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(image, mask, faceBox, bgdModel, fgdModel, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)

    backgrndMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),True,False).astype('uint8')
    backgrndMask = np.broadcast_to(backgrndMask[:,:,np.newaxis], np.shape(image))
    return backgrndMask

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
        backgrndMask = segment(image, faceBoxAdjusted)

    else:
        (x, y, w, h) = faceBoxAdjusted
        backgrndMask = np.full(image.shape, True, dtype=bool)
        backgrndMask[y:y+h, x:x+w, :] = False 
    
    (x, y, w, h) = faceBox
    if REMOVE_EYES:
        backgrndMask[y + h * EYE_LOWER_FRAC : y + h * EYE_UPPER_FRAC, :] = True
    if FOREHEAD_ONLY:
        backgrndMask[y + h * EYE_LOWER_FRAC :, :] = True

    roi = np.ma.array(image, mask=backgrndMask) # Masked array
    return roi

# Sum of square differences between x1, x2, y1, y2 points for each ROI
def distance(roi1, roi2):
    return sum((roi1[i] - roi2[i])**2 for i in range(len(roi1)))

def getBestROI(frame, faceCascade, previousFaceBox):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
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


    # Show rectangle
    #(x, y, w, h) = faceBox
    #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    if faceBox is not None:
        roi = getROI(frame, faceBox)

    return faceBox, roi

def plotSignals(signals, label):
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
    colors = ["r", "g", "b"]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(seconds, signals[:,i], colors[i])
    plt.xlabel('Time (sec)', fontsize=17)
    plt.ylabel(label, fontsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.show()

def plotSpectrum(freqs, powerSpec):
    idx = np.argsort(freqs)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(freqs[idx], powerSpec[idx,i])
    plt.xlabel("Frequency (Hz)", fontsize=17)
    plt.ylabel("Power", fontsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.xlim([0.75, 4])
    plt.show()

def getHeartRate(window, lastHR):
    # Normalize across the window to have zero-mean and unit variance
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std

    # Separate into three source signals using ICA
    ica = FastICA()
    srcSig = ica.fit_transform(normalized)
    #np.save(RESULTS_SAVE_DIR + videoFile[0:-4] + "_ppg", srcSig)

    # Find power spectrum
    powerSpec = np.abs(np.fft.fft(srcSig, axis=0))**2
    freqs = np.fft.fftfreq(WINDOW_SIZE, 1.0 / FPS)

    # Find heart rate
    maxPwrSrc = np.max(powerSpec, axis=1)
    #if lastHR != None:
        #validIdx = np.where((freqs >= lastHR - MAX_HR_CHANGE / SEC_PER_MIN) & (freqs <= lastHR + MAX_HR_CHANGE / SEC_PER_MIN))
    #else: 
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
    validPwr = maxPwrSrc[validIdx]
    validFreqs = freqs[validIdx]
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx]
    print hr

    #plotSignals(normalized, "Normalized color intensity")
    #plotSignals(srcSig, "Source signal strength")
    #plotSpectrum(freqs, powerSpec)

    return hr

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
video = cv2.VideoCapture(VIDEO_DIR + videoFile)
faceCascade = cv2.CascadeClassifier(CASCADE_PATH)

colorSig = [] # Will store the average RGB color values in each frame's ROI
heartRates = [] # Will store the heart rate calculated every 1 second
previousFaceBox = None
while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        break

    previousFaceBox, roi = getBestROI(frame, faceCascade, previousFaceBox)

    if (roi is not None) and (np.size(roi) > 0):
        colorChannels = roi.reshape(-1, roi.shape[-1])
        avgColor = colorChannels.mean(axis=0)
        colorSig.append(avgColor)

    # Calculate heart rate every one second (once have 30-second of data)
    if (len(colorSig) >= WINDOW_SIZE) and (len(colorSig) % np.ceil(FPS) == 0):
        windowStart = len(colorSig) - WINDOW_SIZE
        window = colorSig[windowStart : windowStart + WINDOW_SIZE]
        lastHR = heartRates[-1] if len(heartRates) > 0 else None
        heartRates.append(getHeartRate(window, lastHR))

    if np.ma.is_masked(roi):
        roi = np.where(roi.mask == True, 0, roi)
    cv2.imshow('ROI', roi)
    cv2.waitKey(1)

print heartRates
print videoFile
np.save(RESULTS_SAVE_DIR + videoFile[0:-4], heartRates)
video.release()
cv2.destroyAllWindows()