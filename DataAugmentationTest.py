#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 01:42:27 2019

@author: earendilavari
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:48:51 2019

@author: earendilavari
"""

'''
Only to test, delete after tested
'''
#%%

# Load pickle module
import pickle
import cv2
import numpy as np

trainFile = 'TrainData/train.p'
validationFile = 'TrainData/valid.p'
testFile = 'TrainData/test.p'

with open(trainFile, mode='rb') as file:
    trainData = pickle.load(file)
with open(validationFile, mode='rb') as file:
    validationData = pickle.load(file)
with open(testFile, mode='rb') as file:
    testData = pickle.load(file)
    
trainData_X, trainData_y = trainData['features'], trainData['labels']
validationData_X, validationData_y = validationData['features'], validationData['labels']
testData_X, testData_y = testData['features'], testData['labels']

# It verifies if the images and the labels have the same length
assert(len(trainData_X) == len(trainData_y))
assert(len(validationData_X) == len(validationData_y))
assert(len(testData_X) == len(testData_y))



import numpy as np

# Number of training samples
trainData_quantity = trainData_X.shape[0]

# Number of validation samples
validationData_quantity = validationData_X.shape[0]

# Number of test samples
testData_quantity = testData_X.shape[0]

# Image shape
imagesShape = (trainData_X.shape[1], trainData_X.shape[2], trainData_X.shape[3])

# How many unique classes/labels there are in the dataset.
Labels_quantity = len(np.unique(trainData_y))

print("Number of training examples =", trainData_quantity)
print("Number of testing examples =", testData_quantity)
print("Number of validation examples =", validationData_quantity)
print("Image data shape =", imagesShape)
print("Number of classes =", Labels_quantity)



#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt


# testImage = trainData_X[6475]
testImage = trainData_X[15732]

plt.imshow(testImage)


def rotateImage(image, angle):
    rows, cols = image.shape[:2]
    rotMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotatedImage = cv2.warpAffine(image, rotMatrix, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return rotatedImage

def translateImage(image, translationX, translationY):
    rows, cols = image.shape[:2]
    trMatrix = np.float32([[1,0, translationX], [0,1, translationY]])
    translatedImage = cv2.warpAffine(image, trMatrix, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return translatedImage

def zoomImage(image, zoomFactor):
    rows, cols = image.shape[:2]
    xCenter = cols/2
    yCenter = rows/2
    xLeft = np.int(xCenter/2)
    xRight = np.int(xCenter + (xCenter/2))
    yTop = np.int(yCenter/2)
    yBottom = np.int(yCenter + (yCenter/2))
    
    pointsIn = np.float32([[xLeft, yTop], [xRight, yTop], [xRight, yBottom], [xLeft, yBottom]])
    pointsOut = np.float32([[xLeft-zoomFactor, yTop-zoomFactor], [xRight+zoomFactor, yTop-zoomFactor], 
                            [xRight+zoomFactor, yBottom+zoomFactor], [xLeft-zoomFactor, yBottom+zoomFactor]])
    
    zmMatrix = cv2.getPerspectiveTransform(pointsIn, pointsOut)
    
    zoomedImage = cv2.warpPerspective(image, zmMatrix, (cols, rows), borderMode = cv2.BORDER_REFLECT)
    return zoomedImage


rotatedTestImage = rotateImage(testImage, 20)
translatedTestImage = translateImage(testImage, -10, 10)
zoomedTestImage = zoomImage(testImage, -2)

plt.imshow(rotatedTestImage)
#%%

import random

def augmentData(images, labels, maxTranslationX = 5, maxTranslationY = 5, maxRotation = 15, maxZoom = 4):
    # Creates two lists from the images and labels array in order to be able to stack new images and labels on them
    imagesList = list(images)
    labelsList = list(labels)
    # Gets the length of "imagesList" before any modification in order to avoid an infinite loop
    datasetSize = len(imagesList)
    
    for i in range(datasetSize):
    	augmentationOperation = random.randint(0,2)

    	if augmentationOperation == 0:

        	# Random values to translate
        	translationX = random.randint(-maxTranslationX,maxTranslationX)
        	translationY = random.randint(-maxTranslationX,maxTranslationX)

        	# Creates translated image
        	translatedImage = translateImage(imagesList[i], translationX, translationY)
        	imagesList.append(translatedImage)
        	labelsList.append(labelsList[i])

        elif augmentationOperation == 1:

        	# Appends the image to "imagesList"
        	imagesList.append(translatedImage)

        	# Random value to rotate
        	angleRotation = random.randint(-maxRotation,maxRotation)
        	rotatedImage = rotateImage(imagesList[i], angleRotation)

        	imagesList.append(rotatedImage)
        	labelsList.append(labelsList[i])

        elif augmentationOperation == 2:
        
        	# Random value for zooming
        	zmFactor = random.randint(-maxZoom, maxZoom)
        	# Creates zoomed image
        	zoomedImage = zoomImage(imagesList[i], zmFactor)
        
        	imagesList.append(zoomedImage)
        	labelsList.append(labelsList[i])
        
    # Converts the lists into arrays again
    augmentedImages = np.asarray(imagesList)
    augmentedLabels = np.asarray(labelsList)
    
    return augmentedImages, augmentedLabels