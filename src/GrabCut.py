import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import maxflow

GAMMA = 50
K = 5
NUM_NEIGHBORS = 8

def getPixelArray(pixels3D):
	numPixels = pixels3D.shape[0] * pixels3D.shape[1]
	pixels = np.empty((numPixels, 3))
	for i in range(3):
		pixels[:,i] = pixels3D[:,:,i].flatten()
	return pixels

def getTermWeights(pixels, mask, certainBgMask):
	pixelsInClass = pixels[mask == True]

	GMM = mixture.GMM(n_components=K, covariance_type='full')
	GMM.fit(pixelsInClass)
	membership = GMM.predict(pixelsInClass)

	means = GMM.means_
	covs = GMM.covars_
	invCovs = np.linalg.inv(GMM.covars_)
	sqrtDetCovs = np.sqrt(np.linalg.det(GMM.covars_))
	weights = [float(np.sum(membership == i)) / pixelsInClass.shape[0] for i in range(K)]

	numPixels = pixels.shape[0]
	termWeights = np.empty(numPixels)
	for j in range(numPixels):
		if certainBgMask[j] == False: # Save time by not calculating probability for known background pixels
			sumClusters = 0
			for i in range(K):
				diffMean = pixels[j] - means[i]
				exponent = - 0.5 * diffMean.dot(invCovs[i]).dot(diffMean.T)
				sumClusters += weights[i] * 1.0 / sqrtDetCovs[i] * np.exp(exponent)
			termWeights[j] = -np.log(sumClusters)
	
	return termWeights

def getNewFgBg(height, width, neighborWeightsY, neighborWeightsX, 
	neighborWeightsDiag1, neighborWeightsDiag2, fgWeights, bgWeights):
	numPixels = height * width
	g = maxflow.Graph[float](numPixels, 2 * numPixels * NUM_NEIGHBORS) #num nodes, num non-terminal edges
 	nodeids = g.add_grid_nodes((height, width))

	# Edges pointing up (& down)
	structure = np.zeros((3,3))
	structure[0,1] = 1
	weights = np.vstack((np.zeros((1, width)), neighborWeightsY))
	g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=True)
    
	# Edges pointing left (& right)
	structure = np.zeros((3,3))
	structure[1,0] = 1
	weights = np.hstack((np.zeros((height, 1)), neighborWeightsX))
	g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=True)

	# Edges pointing left-up (& right-down)
	structure = np.zeros((3,3))
	structure[0,0] = 1
	weights = np.hstack((np.zeros((height-1, 1)), neighborWeightsDiag1))
	weights = np.vstack((np.zeros((1, width)), weights))
	g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=True)

	# Edges point right-up (& left-down)
	structure = np.zeros((3,3))
	structure[0,2] = 1
	weights = np.hstack((neighborWeightsDiag2, np.zeros((height-1, 1))))
	weights = np.vstack((np.zeros((1, width)), weights))
	g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=True)

	# Terminal edges
	g.add_grid_tedges(nodeids, fgWeights, bgWeights)

	flow = g.maxflow()
	bgMask = g.get_grid_segments(nodeids)
	fgMask = np.full((height, width), False, dtype=bool)
	fgMask[bgMask == False] = True

	return fgMask, bgMask

def visualizeSegmentation(image, bgMask):
	bgMask3D = np.broadcast_to(bgMask[:,:,np.newaxis], np.shape(image))
	maskedImage = np.ma.array(image, mask=bgMask3D)
	maskedImage[bgMask3D == True] = 0
	cv2.imshow('Segmentation', maskedImage)
	cv2.waitKey(1)

def grabCut(image, box, numIters):
	height = image.shape[0]
	width = image.shape[1]
	numPixels = height * width
	(x, y, w, h) = box

	# Calculate weights between neighboring pixels
	distX = np.sum((image[:,:-1,:] - image[:,1:,:])**2, 2)
	distY = np.sum((image[:-1,:,:] - image[1:,:,:])**2, 2)
	distDiag1 = np.sum((image[:-1,:-1,:] - image[1:,1:,:])**2, 2)
	distDiag2 = np.sum((image[:-1,1:,:] - image[1:,:-1,:])**2, 2)
	neighborWeightsX = GAMMA * np.exp(-0.5 * distX / np.mean(distX))
	neighborWeightsY = GAMMA * np.exp(-0.5 * distY / np.mean(distY))
	neighborWeightsDiag1 = GAMMA/np.sqrt(2) * np.exp(-0.5 * distDiag1 / np.mean(distDiag1))
	neighborWeightsDiag2 = GAMMA/np.sqrt(2) * np.exp(-0.5 * distDiag2 / np.mean(distDiag2))

	# Calculate initial (certainly) background and (potentially) foreground masks
	pixels = getPixelArray(image)
	probFgMask = np.full((height, width), False, dtype=bool)
	probFgMask[y:y+h, x:x+w] = True
	bgMask = np.full((height, width), True, dtype=bool)
	bgMask[y:y+h, x:x+w] = False
	probBgMask = bgMask
	bgMask = bgMask.flatten()

	for i in range(numIters):
		probFgMask = probFgMask.flatten()
		probBgMask = probBgMask.flatten()

		fgWeights = getTermWeights(pixels, probBgMask, bgMask) #FG edge weights depend on likelihood pixel belongs to BG
		fgWeights[bgMask == True] = 0
		fgWeights = fgWeights.reshape((height, width))
		bgWeights = getTermWeights(pixels, probFgMask, bgMask)
		bgWeights[bgMask == True] = NUM_NEIGHBORS*GAMMA + 1
		bgWeights = bgWeights.reshape((height, width))

		probFgMask, probBgMask = getNewFgBg(height, width, neighborWeightsY, neighborWeightsX, 
			neighborWeightsDiag1, neighborWeightsDiag2, fgWeights, bgWeights)

		visualizeSegmentation(image, probBgMask)

	return probFgMask, probBgMask


# grabCut Tests:

#image = cv2.imread("../video/still.jpg")
#box = [396, 171, 220, 331]
#image = cv2.imread("../video/flower.jpeg")
#box = [50, 1, 200, 180]

#(x, y, w, h) = box
#cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
#cv2.imshow('image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#grabCut(image, box, 5)