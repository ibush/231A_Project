import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import maxflow

'''
g = maxflow.Graph[float](2,2) #num nodes, num non-terminal edges
nodes = g.add_nodes(2)
g.add_edge(nodes[0], nodes[1], 1, 2)
g.add_tedge(nodes[0], 2, 5)
g.add_tedge(nodes[1], 9, 4)

flow = g.maxflow()
print flow
print "segment of node 0: ", g.get_segment(nodes[0]) # Returns 1 if w/ source node, 0 if with sink
print "segment of node 1: ", g.get_segment(nodes[1])
'''

BETA = 0.13
K = 5
NUM_NEIGHBORS = 4 # TODO: 8?

# TODO: Have this take in the image and a mask
def getPixelArray(pixels3D):
	numPixels = pixels3D.shape[0] * pixels3D.shape[1]
	pixels = np.empty((numPixels, 3))
	for i in range(3):
		pixels[:,i] = pixels3D[:,:,i].flatten()
	return pixels

# TODO: Don't calculate for pixels that are definitely background
def getTermWeights(pixels, mask):
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
		sumClusters = 0
		for i in range(K):
			diffMean = pixels[j] - means[i]
			exponent = - 0.5 * diffMean.dot(invCovs[i]).dot(diffMean.T)
			sumClusters += weights[i] * 1.0 / sqrtDetCovs[i] * np.exp(exponent)
		termWeights[j] = -np.log(sumClusters)
	
	return termWeights


def grabCut(image, box, numIters):
	height = image.shape[0]
	width = image.shape[1]
	numPixels = height * width
	(x, y, w, h) = box

	distX = np.sum((image[:,:-1,:] - image[:,1:,:])**2, 2)
	distY = np.sum((image[:-1,:,:] - image[1:,:,:])**2, 2)
	neighborWeightsX = np.exp(-BETA * distX)
	neighborWeightsY = np.exp(-BETA * distY)

	pixels = getPixelArray(image)
	probFgMask = np.full((height, width), False, dtype=bool)
	probFgMask[y:y+h, x:x+w] = True
	probFgMask = probFgMask.flatten()
	bgMask = np.full((height, width), True, dtype=bool)
	bgMask[y:y+h, x:x+w] = False
	bgMask = bgMask.flatten()
	probBgMask = bgMask

	for i in range(numIters):
		fgWeights = getTermWeights(pixels, probFgMask)
		fgWeights[bgMask == True] = 0
		fgWeights = fgWeights.reshape((height, width))
		bgWeights = getTermWeights(pixels, probBgMask)
		bgWeights[bgMask == True] = 1 # TODO: Fancier calculation in paper
		bgWeights = bgWeights.reshape((height, width))

		g = maxflow.Graph[float](numPixels, 2 * numPixels * NUM_NEIGHBORS) #num nodes, num non-terminal edges


 		nodeids = g.add_grid_nodes((height, width))
		# Edges pointing up 
		structure = np.zeros((3,3))
		structure[0,1] = 1
		weights = np.vstack((np.zeros((1, width)), neighborWeightsY))
		g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)
    
    	# TODO: Set symmetric=True and delete down & right
		# Edges pointing down
		structure = np.zeros((3,3))
		structure[2,1] = 1
		weights = np.vstack((neighborWeightsY, np.zeros((1, width))))
		g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

		# Edges pointing left
		structure = np.zeros((3,3))
		structure[1,0] = 1
		weights = np.hstack((np.zeros((height, 1)), neighborWeightsX))
		g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

		# Edges pointing right
		structure = np.zeros((3,3))
		structure[1,2] = 1
		weights = np.hstack((neighborWeightsX, np.zeros((height, 1))))
		g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

		# Terminal edges
		g.add_grid_tedges(nodeids, fgWeights, bgWeights)

		flow = g.maxflow()
		probFgMask = g.get_grid_segments(nodeids)
		probBgMask = np.full((height, width), True, dtype=bool)
		probBgMask[probFgMask == True] = False

		backgrndMask = np.broadcast_to(probBgMask[:,:,np.newaxis], np.shape(image))
		maskedImage = np.ma.array(image, mask=backgrndMask) # Masked array
		maskedImage = np.where(backgrndMask == True, 0, maskedImage)
		cv2.imshow('Segmentation', maskedImage)
		cv2.waitKey(1)

		probFgMask = probFgMask.flatten()
		probBgMask = probBgMask.flatten()

'''
		nodes = g.add_grid_nodes(numPixels)
		for i in range(height - 1):
			for j in range(width - 1):
				ind = i * width + j
				g.add_edge(nodes[ind], nodes[ind+1], neighborWeightsX[i][j], neighborWeightsX[i][j])
				g.add_edge(nodes[ind], nodes[ind+width], neighborWeightsY[i][j], neighborWeightsY[i][j])
		for i in range(numPixels):
			g.add_tedge(nodes[i], fgWeights[i], bgWeights[i])
'''


'''
	clusters = []
	C1 = np.mgrid[y:y+h, x:x+w].reshape(2, -1).T
	print C1.shape

	clusters.append(C1)
	for i in range(K):
		maxEigVal = 0
		maxEigVec = []
		maxInd = 0
		for j in range(i+1):
			jClusterVals = image[clusters[j][:,0], clusters[j][:,1], :]
			eigval, eigvec = np.linalg.eig(jClusterVals)
			if np.max(eigval) > maxEigVal:
				maxEigVal = np.max(eigval)
				maxEigVec = eigvec[np.argmax(eigval,1)]
				maxInd = j

		clusterVals = image[clusters[maxInd][:,0], clusters[maxInd][:,1], :]
		threshold = maxEigVec.T * np.mean(clusterVals)
		split1 = clusters[i][np.where(maxEigVec.T * clusterVals >= threshold)]
		split2 = clusters[i][np.where(maxEigVec.T * clusterVals < threshold)]
		clusters[i] = split1
		clusters.append(split2)
'''
image = cv2.imread("../video/still.jpg")
box = [396, 171, 220, 331]
#(x, y, w, h) = box
#cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
#cv2.imshow('image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

grabCut(image, box, 5)