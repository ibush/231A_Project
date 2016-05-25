import numpy as np
import matplotlib.pyplot as plt
import sys

SPS = 200
MIN_HR_BPM = 45.0
MAX_HR_BMP = 240.0
SEC_PER_MIN = 60
WINDOW_TIME_SEC = 30
WINDOW_SIZE = WINDOW_TIME_SEC * SPS

RESULTS_SAVE_DIR = "../results/reference/" 
DEFAULT_PPG_FILE = "true_HR_2.txt"

def plotSignal(signal):
    seconds = np.arange(0, len(signal) / SPS, 1.0 / SPS)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.plot(seconds, signal[0:len(seconds)])
    plt.gca().get_yaxis().set_visible(False)
    plt.ylabel("PPG signal", fontsize=17)
    plt.xlabel("Time (sec)", fontsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.show()

def plotPowerSpectrum(freqs, powerSpec):
    idx = np.argsort(freqs)
    plt.figure()
    plt.plot(freqs[idx], powerSpec[idx])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim([0.75, 4])
    plt.show()

def calcHR(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    normalized = (signal - mean) / std

    # Find power spectrum
    powerSpec = np.abs(np.fft.fft(normalized))**2
    freqs = np.fft.fftfreq(len(signal), 1.0 / SPS)

    # Find heart rate
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
    validPwr = powerSpec[validIdx]
    validFreqs = freqs[validIdx]
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx]
    return hr
    #plotPowerSpectrum(freqs, powerSpec)


try:
    ppgFile = sys.argv[1]
except:
    ppgFile = DEFAULT_PPG_FILE  

print ppgFile

signal = np.fromfile(ppgFile, sep=' ')
plotSignal(signal[600:1600])

heartRates = []
for i in range(0, len(signal) - WINDOW_SIZE, SPS):
    window = signal[i : i + WINDOW_SIZE]
    heartRates.append(calcHR(window))

print heartRates
np.save(RESULTS_SAVE_DIR + ppgFile[0:-4], heartRates)
np.save(RESULTS_SAVE_DIR + ppgFile[0:-4] + "_ppg", signal)