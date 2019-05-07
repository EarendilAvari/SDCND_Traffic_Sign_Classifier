#%% 
import cv2
import numpy as np
import random

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

def augmentData(images, labels, maxTranslationX = 5, maxTranslationY = 5, maxRotation = 15, maxZoom = 4):
    # Creates two lists from the images and labels array in order to be able to stack new images and labels on them
    imagesList = list(images)
    labelsList = list(labels)
    # Gets the length of "imagesList" before any modification in order to avoid an infinite loop
    datasetSize = len(imagesList)
    
    for i in range(datasetSize):
        augmentationOperation = random.randint(0,4)

        if (augmentationOperation == 0) | (augmentationOperation == 1):

            # Random values to translate
            translationX = random.randint(-maxTranslationX,maxTranslationX)
            translationY = random.randint(-maxTranslationX,maxTranslationX)
            # Creates translated image
            translatedImage = translateImage(imagesList[i], translationX, translationY)
            # Appends image and label to lists
            imagesList.append(translatedImage)
            labelsList.append(labelsList[i])

        elif (augmentationOperation == 2) | (augmentationOperation == 3):

            # Random value to rotate
            angleRotation = random.randint(-maxRotation,maxRotation)
            # Creates rotated image
            rotatedImage = rotateImage(imagesList[i], angleRotation)
            # Appends image and label to lists
            imagesList.append(rotatedImage)
            labelsList.append(labelsList[i])

        elif augmentationOperation == 4:
        
            # Random value for zooming
            zmFactor = random.randint(-maxZoom, maxZoom)
            # Creates zoomed image
            zoomedImage = zoomImage(imagesList[i], zmFactor)
            # Appends image and label to lists        
            imagesList.append(zoomedImage)
            labelsList.append(labelsList[i])
        
    # Converts the lists into arrays again
    augmentedImages = np.asarray(imagesList)
    augmentedLabels = np.asarray(labelsList)
    
    return augmentedImages, augmentedLabels


