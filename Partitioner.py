import cv2
import numpy as np

class Partitioner:
	'''
	This class is used to Partition the data into training and testing data.
	'''
	def __init__(self):
		self.list_of_images = []
		self.list_of_labels = []
		self.train_data_x = []
		self.train_target_y = []
		self.test_data_x = []
		self.test_target_y = []
		self.base_dir_name = ''
		self.class_name = ''
		self.total_classes = 0
		self.total_sample = 0
		self.img_extension = ''
		self.total_samples_overall = 0
		self.training_samples_number = 0
		self.testing_samples_number = 0

	def readData(self,base_dir_name,class_name,total_classes,total_sample,img_extension):
		'''
		This module is used to read the data(images) from the directory.

		#Parameters: 'base_dir_name' is the name of the directory containing all the data,
					 'class_name' is common name for all the directories containing classified data,
					 'total_classes' is the total number of classifications,
					 'total_sample' is the total number of samples in each classification. 
					 'img_extension' is the image extension like .jpg , .png etc.
		#Return: List of images(read data) 
		'''
		self.base_dir_name = base_dir_name
		self.class_name = class_name
		self.total_classes = total_classes
		self.total_sample = total_sample
		self.img_extension = img_extension

		for j in range(total_sample):
			for i in range(total_classes):
				img = cv2.imread(base_dir_name + '/' + class_name + str(i+1) + '/' + str(j+1) + img_extension,0)
				img = np.asarray(img)
				img = img.flatten()
				self.list_of_images.append(img)
				self.list_of_labels.append(i)
		return self.list_of_images

	def splitDataSet(self,training_percent):
		'''
		This module is used to split the data into training and testing data.

		#Parameters: 'training_percent' is the percentage of data to be used for training.
		#Return: None.
		'''
		self.total_samples_overall = len(self.list_of_images)
		self.training_samples_number = int(training_percent*self.total_samples_overall)
		self.testing_samples_number = int(self.total_samples_overall - self.training_samples_number)
		for i in range(self.training_samples_number):
			self.train_data_x.append(self.list_of_images[i])
			self.train_target_y.append(self.list_of_labels[i])
		for j in range(self.training_samples_number,self.total_samples_overall):
			self.test_data_x.append(self.list_of_images[j])
			self.test_target_y.append(self.list_of_labels[j])

	def getTrainingData(self):
		'''
		This module is used to get the data which is being used for training.

		#Parameters: None
		#Return: List of training data
		'''
		self.train_data_x = np.asarray(self.train_data_x)
		return self.train_data_x

	def getTrainingLabel(self):
		'''
		This module is used to get the labels which are being used for training.

		#Parameters: None
		#Return: List of training labels
		'''
		self.train_target_y = np.asarray(self.train_target_y)
		return self.train_target_y

	def getTestingData(self):
		'''
		This module is used to get the data which is being used for testing.

		#Parameters: None
		#Return: List of testing data
		'''
		self.test_data_x = np.asarray(self.test_data_x)
		return self.test_data_x

	def getTestingLabel(self):
		'''
		This module is used to get the labels which are being used for testing.

		#Parameters: None
		#Return: List of testing labels
		'''
		self.test_target_y = np.asarray(self.test_target_y)
		return self.test_target_y


			


