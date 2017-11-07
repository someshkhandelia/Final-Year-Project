import cv2
import numpy as np
import os

class PGMConverter:
    '''
    This class is used to convert other image formats to .pgm
    '''

    def __init__(self,base_dir_name,class_name,total_classes,total_sample,img_extension):
        self.base_dir_name = base_dir_name
        self.class_name = class_name
        self.total_classes = total_classes
        self.total_sample = total_sample
        self.img_extension = img_extension

    def createDirectories(self):
        '''
        This module is used to create directories for training data(.pgm).

		#Parameters: None
		#Return: None
        '''
        for i in range(self.total_classes):
            file_path = 'TrainingData' + '/' + self.class_name + str(i+1) + '/'
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

    def convertToPGM(self):
        '''
        This module is used to read sample data and convert it to .pgm.

		#Parameters: None
		#Return: None
        '''
        for j in range(self.total_sample):
            for i in range(self.total_classes):
                img = cv2.imread(self.base_dir_name + '/' + self.class_name + str(i+1) + '/' + str(j+1) + self.img_extension,0)
                dst = cv2.resize(img,(600,600), 0, 0);
                cv2.imwrite('TrainingData' + '/' + self.class_name + str(i+1) + '/' + str(j+1) + '.pgm',dst)
