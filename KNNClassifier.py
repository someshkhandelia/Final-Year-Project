import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

class KNNClassifier:
	'''
	This class is used to train and test our K nearest neighbors classifier.
	'''

	def __init__(self):
		self.KNN_classifier = KNeighborsClassifier()
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
		self.train_images = train_data_x
		self.KNN_classifier.fit(train_data_x,train_target_y)

	def testClassifier(self,test_data_x,test_target_y):
		'''
		This module is used to test our classifier.

		#Parameters: 'test_data_x' is the testing image samples,
					 'test_target_y' is the testing labels for the testing data.
		#Return: Accuracy of the classifier.
		'''
		self.test_images = test_data_x
		self.predicted_results = self.KNN_classifier.predict(self.test_images)
		return accuracy_score(test_target_y,self.predicted_results)

	def saveClassifier(self,pickle_name):
		'''
		This module is to save the trained classifier as a pickle file to be reused.

		#Parameters: 'pickle_name' is the name of the pickle file to be saved as.
		#Return: None
		'''
		with open(pickle_name + '.pkl' , 'wb') as fid:
			pickle.dump(self.KNN_classifier,fid)

	def loadClassifier(self,pickle_name):
		'''
		This module is to load the trained classifier from a pickle file.

		#Parameters: 'pickle_name' is the name of the pickle file to be loaded.
		#Return: KNN_classifier object loaded from the stored pickle
		'''
		with open(pickle_name + '.pkl' , 'rb') as fid:
			self.KNN_classifier = pickle.load(fid)
		return self.KNN_classifier




