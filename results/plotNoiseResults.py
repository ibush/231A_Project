import numpy as np
import matplotlib.pyplot as plt

FILE = "results_noise.csv"

box = []
segment = []

with open(FILE) as f:
	for line in f:
		if line[0] == 'n':
			box.append(line.split(','))
		else:
			segment.append(line.split(','))

box = np.array(box)
segment = np.array(segment)

# Plot percent outliers vs. noise
noisePercent = range(0, 55, 5)
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(noisePercent, box[:, 6].astype(float) * 100, label="Box")
plt.plot(noisePercent, segment[:, 6].astype(float) * 100, label="Segmentation")
plt.xlabel("Bounding box noise (%)", fontsize=17)
plt.ylabel("Outliers (%)", fontsize=17)
plt.tick_params(axis='x', labelsize=17)
plt.tick_params(axis='y', labelsize=17)
plt.legend(loc='lower right', prop={'size': 17})
plt.show()