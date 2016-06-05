import sys
import numpy as np
import matplotlib.pyplot as plt

VIDEO_TIME_SEC = 60
WINDOW_TIME_SEC = 30
NUM_MEASUREMENTS = VIDEO_TIME_SEC - WINDOW_TIME_SEC
VIDEO_START_OFFSET = 2
MIN_PER_SEC = 60
SPS = 200
FPS = 15
PLOT_LEN_SEC = 5

def plotHeartRates(ref, measured):
	time = range(NUM_MEASUREMENTS)
	plt.figure()
	plt.plot(time, ref, time, measured)
	plt.xlabel("Time (sec)")
	plt.ylabel("Heart rate")
	plt.show()

def plotPPG(ref, measured):
	secondsRef = np.arange(0, PLOT_LEN_SEC, 1.0 / SPS)
	secondsMeasured = np.arange(0, PLOT_LEN_SEC, 1.0 / FPS)

    # Normalize reference
	mean = np.mean(ref)
	std = np.std(ref)
	refNorm = (ref - mean) / std
    # Normalize measured
	mean = np.mean(measured)
	std = np.std(measured)
	measuredNorm = (measured - mean) / std

	fig = plt.figure()
	fig.patch.set_facecolor('white')
	plt.plot(secondsRef, refNorm[len(secondsRef):2*len(secondsRef)], label="Reference") 
	plt.plot(secondsMeasured, measuredNorm[0:len(secondsMeasured)], label="ICA Source")
	plt.xlabel('Time (sec)', fontsize=17)
	plt.ylabel('Normalized PPG signal', fontsize=17)
	plt.tick_params(axis='x', labelsize=17)
	plt.tick_params(axis='y', labelsize=17)
	plt.legend(loc='lower right', prop={'size': 17})
	plt.show()

refFile = sys.argv[1]
dataFile = sys.argv[2]
refHR = np.load(refFile)
measuredHR = np.load(dataFile)

try:
	refPPG = np.load(refFile[0:-4] + "_ppg.npy")
	dataPPG = np.load(dataFile[0:-4] + "_ppg.npy")
	plotPPG(refPPG, dataPPG[:, 1]) # Change for different source signals (0-2)
except:
	pass

# Cut all measurements at 1 minute
refHR = refHR[VIDEO_START_OFFSET : VIDEO_START_OFFSET + NUM_MEASUREMENTS] * MIN_PER_SEC
measuredHR = measuredHR[VIDEO_START_OFFSET : VIDEO_START_OFFSET + NUM_MEASUREMENTS] * MIN_PER_SEC
errorBPM = refHR - measuredHR
errorPercent = np.abs(errorBPM) / refHR * 100
percentOutliers = float(np.sum(errorPercent > 10)) / len(errorPercent)
inlierErrorBPM = errorBPM[errorPercent <= 10]

#print measuredHR
#print errorBPM
#print errorPercent

with open("results.csv", "a") as f:
	f.write(dataFile + "," 
		+ str(np.mean(errorBPM)) + "," + str(np.std(errorBPM)) + "," 
		+ str(np.mean(errorPercent)) + "," + str(np.std(errorPercent)) + "," 
		+ str(percentOutliers) + "," 
		+ str(np.mean(inlierErrorBPM)) + "," + str(np.std(inlierErrorBPM)) + "\n")

print "Avg error BPM: " + str(np.mean(errorBPM))
print "Std dev BPM: " + str(np.std(errorBPM))
print "Avg error percent: " + str(np.mean(errorPercent))
print "Std dev percent: " + str(np.std(errorPercent))
print "Percent outliers: " + str(percentOutliers)

#plotHeartRates(refHR, measuredHR)