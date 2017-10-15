import cv2
from tensorflow.contrib import learn
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import pickle

class NeuralNetwork:
	'''
	This class is used to train and test our K nearest neighbors classifier.
	'''

	def __init__(self):
		self.feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
		self.DNN_classifier = learn.DNNClassifier(feature_columns=self.feature_columns,hidden_units=[10,20,10],n_classes=40)
		self.train_images = []
		self.test_images = []
		self.predicted_results = []

	def trainClassifier(self,train_data_x,train_target_y):
		'''
		This module is used to train our classifier.

		#Parameters: 'train_data_x' is the training image samples,
					 'train_target_y' is the training labels for the training data. 
		#Return: None
		'''		
		train_data_x = np.asarray(train_data_x)
		train_target_y = np.asarray(train_target_y)
		self.train_images = train_data_x
		self.DNN_classifier.fit(train_data_x, train_target_y, steps = 200)

	def testClassifier(self,test_data_x,test_target_y):
		'''
		This module is used to test our classifier.

		#Parameters: 'test_data_x' is the testing image samples,
					 'test_target_y' is the testing labels for the testing data.
		#Return: Accuracy of the classifier.
		'''
		test_data_x = np.asarray(test_data_x)
		self.test_images = test_data_x
		test_target_y = np.asarray(test_target_y)
		self.predicted_results = self.DNN_classifier.predict(self.test_images)
		print("Actual:")
		print(test_target_y)
		print("Predicted:")
		print(list(self.predicted_results))
		self.predicted_results = np.asarray(list(self.predicted_results))
		#print('accuracy score: ')
		print(accuracy_score(test_target_y,self.predicted_results))

	def saveClassifier(self,pickle_name):
		'''
		This module is to save the trained classifier as a pickle file to be reused.

		#Parameters: 'pickle_name' is the name of the pickle file to be saved as.
		#Return: None
		'''
		with open(pickle_name + '.pkl' , 'wb') as fid:
			pickle.dump(self.DNN_classifier,fid)	

	def loadClassifier(self,pickle_name):
		'''
		This module is to load the trained classifier from a pickle file.

		#Parameters: 'pickle_name' is the name of the pickle file to be loaded.
		#Return: None
		'''
		with open(pickle_name + '.pkl' , 'rb') as fid:
			self.DNN_classifier = pickle.load(fid)




