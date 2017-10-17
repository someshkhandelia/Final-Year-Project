import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from KNNClassifier import KNNClassifier
from TreeClassifier import TreeClassifier
import numpy as np

class ImageFeederKNN:
	'''
	This class is used to feed the images of faces to the classifier,
	and getting back the predictions.
	This should be used after the classifier has been trained.
	'''
	
	def __init__(self):
		self.KNN_classifier = KNeighborsClassifier()
		self.KNN_obj = KNNClassifier()
		self.predicted_results = []
		self.array_of_images = []
		
	def convertRawDataToTestData(self,list_of_images):
		'''
		This module is used to convert the list of images,
		into a format that can be accepted by the classifier.
		This should be called before getPrediction().

		#Parameters: 'list_of_images' is the List of images for which predictions are needed.
		#Return: None

		'''
		length_of_list = len(list_of_images)
		for i in range(length_of_list):
			img = list_of_images[i]
			img = np.asarray(img)
			img = img.flatten()
			self.array_of_images.append(img)
		self.array_of_images = np.asarray(self.array_of_images)

	def getPrediction(self,pickle_name):
		'''
		This module loads the classifier and,
		predicts the label of the images passed to it.

		#Parameters: 'pickle_name' is the pickle file to be loaded
		#Return: Array of predictions
		'''
		self.KNN_classifier = self.KNN_obj.loadClassifier(pickle_name)
		self.predicted_results = self.KNN_classifier.predict(self.array_of_images)
		return self.predicted_results

