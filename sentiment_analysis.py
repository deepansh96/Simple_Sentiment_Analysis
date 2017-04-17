from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb # Load pre-processed IMDB movie reviews 
								  # from tflearn datasets


# IMDB Dataset loading 
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, #converting to .pkl extension
                                valid_portion=0.1) 				#as it's easy to operate upon 
																#in python.

																#n_words = 10000, we're 
																#extracting 10k words from 
																# the reviews

																#portion of data for validation
																#set is 1%


trainX, trainY = train 				# saving the training data into input(X) and target(Y) vectors 											
testX, testY = test 				# saving the test data into input(X) and target(Y) vectors	



# Data preprocessing


# Sequence padding- We need to pad sequences to perform seq-to-seq prediction.
# Using the pad_sequences function of tflearn
trainX = pad_sequences(trainX, maxlen=100, value=0.) #one seq should be of max length 100
testX = pad_sequences(testX, maxlen=100, value=0.)	 #and padded with values '0'


# Converting labels to binary vectors				 #converting to binary vectors as there are
trainY = to_categorical(trainY, nb_classes=2)		 #two classes- 0(-ve review),1(+ve review)
testY = to_categorical(testY, nb_classes=2)			 #and the computer can't understand labels

# Network building													# Building a five layer deep
net = tflearn.input_data([None, 100])								# neural net
net = tflearn.embedding(net, input_dim=10000, output_dim=128)		
net = tflearn.lstm(net, 128, dropout=0.8)							#LSTM for long-term dependencies
net = tflearn.fully_connected(net, 2, activation='softmax')			
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)			#training using the tflearn Deep Neural Net module
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)