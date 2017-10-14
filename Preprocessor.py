import cv2
import numpy as np

class Preprocessor:
	'''
	This class is to preprocess the captured image
	'''

	def __init__(self):
		pass

	def increaseBrightness(self,img):
		'''
        This module increases the brightness of the captured Image.
        
        #Parameters: Captured Image.
        #Return : Image with increased brightness.
        
        '''
		row,col,channels = img.shape
		temp = img
		for i  in range(row):
			for j in range(col):
				for k in range(channels):
					if temp[i,j,k] + 30 <= 255 :
						temp[i,j,k] = temp[i,j,k] + 30
					else:
						temp[i,j,k]  = 255
		return temp


	def decreaseBrightness(self,img):
		'''
        This module decreases the brightness of the captured Image.
        
        #Parameters: Captured Image.
        #Return : Image with decreased brightness.
        
        '''
		row,col,channels = img.shape
		temp = img
		for i  in range(row):
			for j in range(col):
				for k in range(channels):
					if temp[i,j,k] - 30 >= 0 :
						temp[i,j,k] = temp[i,j,k] - 30
					else:
						temp[i,j,k]  = 0
		return temp
		

	def adjustContrast(self,img):
		'''
        This module increases the contrast of the captured Image,
        by performing Histogram Equalization.
        
        #Parameters: Captured Image.
        #Return : Image with increased contrast.
        
        '''
		img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		# equalize the histogram of the Y channel
		img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
		# convert the YUV image back to RGB format
		img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)		
		return img_output