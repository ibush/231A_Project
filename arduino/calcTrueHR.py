import numpy as np
import matplotlib.pyplot as plt

SPS = 200
MIN_HR_BPM = 45.0
MAX_HR_BMP = 240.0
SEC_PER_MIN = 60

def plotSignal(signal):
    seconds = np.arange(0, len(signal) / SPS, 1.0 / SPS
    plt.figure()
    plt.plot(seconds, signal[0:len(seconds)])
    plt.xlabel("Time (sec)")
    plt.ylabel("PPG signal")
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
    print hr

    # Plot power spectrum
    idx = np.argsort(freqs)
    plt.figure()
    plt.plot(freqs[idx], powerSpec[idx])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim([0.75, 4])
    plt.show()

signal = np.fromfile("true_HR.txt", sep=' ')
plotSignal(signal)
calcHR(signal)