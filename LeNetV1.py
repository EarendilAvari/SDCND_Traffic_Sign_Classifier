import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

def LeNet_V1(x, keepProbDropout):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    wc1 = tf.Variable(tf.truncated_normal(shape = [5,5,3,6], mean = mu, stddev = sigma))
    bc1 = tf.Variable(tf.zeros(6))
    stridesc1 = [1,1,1,1]
    conv1mat = tf.nn.conv2d(x, wc1, stridesc1, padding = 'VALID')
    conv1out = tf.nn.bias_add(conv1mat, bc1)
    
    # TODO: Activation.
    conv1relu = tf.nn.relu(conv1out)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    stridesp1 = [1,2,2,1]
    ksizep1 = [1,2,2,1]
    pool1out = tf.nn.max_pool(conv1relu, ksizep1, stridesp1, padding = 'SAME')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    wc2 = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    bc2 = tf.Variable(tf.zeros(16))
    stridesc2 = [1,1,1,1]
    conv2mat = tf.nn.conv2d(pool1out, wc2, stridesc2, padding = 'VALID')
    conv2out = tf.nn.bias_add(conv2mat, bc2)
    
    # TODO: Activation.
    conv2relu = tf.nn.relu(conv2out)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    stridesp2 = [1,2,2,1]
    ksize2 = [1,2,2,1]
    pool2out = tf.nn.max_pool(conv2relu, ksize2, stridesp2, padding = 'SAME')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    convFlatten = tf.contrib.layers.flatten(pool2out)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    wfc1 = tf.Variable(tf.truncated_normal(shape = [400, 120], mean = mu, stddev = sigma))
    bfc1 = tf.Variable(tf.zeros(120))
    fc1mat = tf.matmul(convFlatten, wfc1)
    fc1out = tf.add(fc1mat, bfc1)
    
    # TODO: Activation.
    fc1relu = tf.nn.relu(fc1out)
    
    # Dropout 
    fc1dropout = tf.nn.dropout(fc1relu, keepProbDropout)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    wfc2 = tf.Variable(tf.truncated_normal(shape = [120, 84], mean = mu, stddev = sigma))
    bfc2 = tf.Variable(tf.zeros(84))
    fc2mat = tf.matmul(fc1dropout, wfc2)
    fc2out = tf.add(fc2mat, bfc2)
    
    # TODO: Activation.
    fc2relu = tf.nn.relu(fc2out)
    
    # Dropout
    fc2dropout = tf.nn.dropout(fc2relu, keepProbDropout)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    wfc3 = tf.Variable(tf.truncated_normal(shape = [84, 43], mean = mu, stddev = sigma))
    bfc3 = tf.Variable(tf.zeros(43))
    fc3mat = tf.matmul(fc2dropout, wfc3)
    fc3out = tf.add(fc3mat, bfc3)
    
    logits = fc3out
    
    return logits

def LeNet_Improved(x, keepProbDropout):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 32x32x10.
    wc1 = tf.Variable(tf.truncated_normal(shape = [5,5,3,10], mean = mu, stddev = sigma))
    bc1 = tf.Variable(tf.zeros(10))
    stridesc1 = [1,1,1,1]
    conv1mat = tf.nn.conv2d(x, wc1, stridesc1, padding = 'SAME')
    conv1out = tf.nn.bias_add(conv1mat, bc1)
    
    # TODO: Activation.
    conv1relu = tf.nn.relu(conv1out)

    # TODO: Pooling. Input = 32x32x10. Output = 16x16x10.
    stridesp1 = [1,2,2,1]
    ksizep1 = [1,2,2,1]
    pool1out = tf.nn.max_pool(conv1relu, ksizep1, stridesp1, padding = 'SAME')
    
    # TODO: Layer 2: Convolutional. Input = 16x16x10, Output = 16x16x18.
    wc2 = tf.Variable(tf.truncated_normal(shape = [4,4,10,18], mean = mu, stddev = sigma))
    bc2 = tf.Variable(tf.zeros(18))
    stridesc2 = [1,1,1,1]
    conv2mat = tf.nn.conv2d(pool1out, wc2, stridesc2, padding = 'SAME')
    conv2out = tf.nn.bias_add(conv2mat, bc2)
    
    # TODO: Activation.
    conv2relu = tf.nn.relu(conv2out)

    # TODO: Pooling. Input = 16x16x18. Output = 8x8x18.
    stridesp2 = [1,2,2,1]
    ksize2 = [1,2,2,1]
    pool2out = tf.nn.max_pool(conv2relu, ksize2, stridesp2, padding = 'SAME')

    # TODO: Layer 3: Convolutional. Input = 8x8x18, Output = 6x6x30.
    wc3 = tf.Variable(tf.truncated_normal(shape = [3,3,18,30], mean = mu, stddev = sigma))
    bc3 = tf.Variable(tf.zeros(30))
    stridesc3 = [1,1,1,1]
    conv3mat = tf.nn.conv2d(pool2out, wc3, stridesc3, padding = 'VALID')
    conv3out = tf.nn.bias_add(conv3mat, bc3)

    # Activation 
    conv3relu = tf.nn.relu(conv3out)

    # Flatten output of layer 3 from 6x6x30 to 1080
    convFlatten = tf.contrib.layers.flatten(conv3relu)
    
    # TODO: Layer 4: Fully Connected. Input = 1080. Output = 490.
    wfc1 = tf.Variable(tf.truncated_normal(shape = [1080, 490], mean = mu, stddev = sigma))
    bfc1 = tf.Variable(tf.zeros(490))
    fc1mat = tf.matmul(convFlatten, wfc1)
    fc1out = tf.add(fc1mat, bfc1)
    
    # TODO: Activation.
    fc1relu = tf.nn.relu(fc1out)
    
    # Dropout 
    fc1dropout = tf.nn.dropout(fc1relu, keepProbDropout)

    # TODO: Layer 5: Fully Connected. Input = 490. Output = 220.
    wfc2 = tf.Variable(tf.truncated_normal(shape = [490, 220], mean = mu, stddev = sigma))
    bfc2 = tf.Variable(tf.zeros(220))
    fc2mat = tf.matmul(fc1dropout, wfc2)
    fc2out = tf.add(fc2mat, bfc2)
    
    # TODO: Activation.
    fc2relu = tf.nn.relu(fc2out)
    
    # Dropout
    fc2dropout = tf.nn.dropout(fc2relu, keepProbDropout)

    # TODO: Layer 6: Fully Connected. Input = 220. Output = 43.
    wfc3 = tf.Variable(tf.truncated_normal(shape = [220, 43], mean = mu, stddev = sigma))
    bfc3 = tf.Variable(tf.zeros(43))
    fc3mat = tf.matmul(fc2dropout, wfc3)
    fc3out = tf.add(fc3mat, bfc3)

    # Logits
    logits = fc3out
    
    return logits

def trainNetwork (trainImages, trainLabels, validImages, validLabels, trainingTensor, lossTensor, accuracyTensor, X_ph, Y_ph, keepProb_ph,
				  epochs = 10, batchSize = 128, dropoutKeepProb = 0.7):

	# Empty lists with data to be plotted
	batchesList = []
	lossList = []
	trainingAccList = []
	validationAccList = []

	batchCount = int(np.ceil(len(trainImages)/batchSize))
	session = tf.get_default_session()

	for epoch_i in range(epochs):
		# Shuffles the data so it does not make the same sequence for every epoch
		trainImages, trainLabels = shuffle(trainImages, trainLabels)

		# Progress bar:
		barDesc = 'Epoch {:>2}/{}'.format(epoch_i+1, epochs)
		batchesPbar = tqdm(range(batchCount), desc= barDesc, unit='batches')

		# Initializes training accuracy at start of every batch
		trainingAccBatchSum = 0

		# Training cycle 
		for batch_i in batchesPbar:
			# Get the batch of training images and labels
			batchStart = batch_i*batchSize
			batchEnd = batchStart + batchSize
			batchTrainImages = trainImages[batchStart:batchEnd]
			batchTrainLabels = trainLabels[batchStart:batchEnd]

			# Runs the optimizer and saves the loss on batchLoss
			trainingFeedDict = {X_ph: batchTrainImages, Y_ph: batchTrainLabels, keepProb_ph: dropoutKeepProb}
			_, batchLoss = session.run([trainingTensor, lossTensor], feed_dict = trainingFeedDict)

			# Calculates accuracy of the current batch in order to get training accuracy
			trainingAccFeedDict = {X_ph: batchTrainImages, Y_ph: batchTrainLabels, keepProb_ph: 1.0}
			batchTrainAcc = session.run(accuracyTensor, feed_dict = trainingAccFeedDict)
			trainingAccBatchSum += batchTrainAcc

			# Logs every 50 batches
			logBatchStep = 50
			if not ((epoch_i*len(batchesPbar) + batch_i) % logBatchStep):
				# Calculates training accuracy based on the executed batches
				lastTrainAcc = trainingAccBatchSum / (batch_i + 1)

				# Calculates validation accuracy
				validationAccFeedDict = {X_ph: validImages, Y_ph: validLabels, keepProb_ph: 1.0}
				lastValidAcc = session.run(accuracyTensor, feed_dict = validationAccFeedDict)
				batchesPbar.set_description('Epoch {:>2}/{} | Loss: {:.2f} | Train Acc.: {:.2f} | Val. Acc.: {:.2f}'
                                            .format(epoch_i+1, epochs, batchLoss, lastTrainAcc, lastValidAcc))
				batchesPbar.refresh()

				# Logs data in order to plot it afterwards:
				batchesList.append(epoch_i*len(batchesPbar) + batch_i)
				lossList.append(batchLoss)
				trainingAccList.append(lastTrainAcc)
				validationAccList.append(lastValidAcc)

	return batchesList, lossList, trainingAccList, validationAccList




