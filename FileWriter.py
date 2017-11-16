import datetime
import os
import numpy as np

class FileWriter:
	'''
	This class is to write the predicted results to a file.
	We shall maintain separate folders for each course.
	'''

	def __init__(self,course_code):
		self.root_folder = 'Attendance'
		self.course_code = course_code
		self.curr_date = datetime.datetime.now().strftime ("%d-%m-%Y")
		self.file_path = self.root_folder + '/' + self.course_code + '/' + self.curr_date + '.txt'

	def saveToFile(self,predictions):
		'''
		This module is to write the predictions to a file.

		#Parameters: 'predictions' is the list of predictions by the classifier.
		#Return: None.
		'''
		directory = os.path.dirname(self.file_path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		predictions = np.asarray(predictions)
		predictions = np.sort(predictions)
		predictions = set(predictions)
		f = open(self.file_path, 'w')
		for item in predictions:
			f.write(str(item) + '\n')
		f.close()


		
     	
     		    		
    	
    	
    	
		
			
		
		

		




        


