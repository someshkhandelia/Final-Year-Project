import cv2

class FaceDetector:
	'''
	This class is to detect faces in a captured image.
	'''

	def __init__(self,img):
		self.face_cascade = cv2.CascadeClassifier('HaarCascades/haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier('HaarCascades/haarcascade_eye.xml')
		self.img = img
		self.gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		self.number_of_detected_faces = 0

	def detectFacesInImage(self):
		'''
        This module detects faces in a captured image.
        It prints the original captured image with rectangles around the faces detected.
        
        #Parameters: None.
        #Return : List of detected faces.
        
        '''
		faces = self.face_cascade.detectMultiScale(self.gray,1.3,5)
		list_of_faces = []
		for (x,y,w,h) in faces:	
		    cv2.rectangle(self.img,(x,y),(x+w,y+h),(255,0,0),2)
		    roi_face = self.gray[y:y+h,x:x+w]
		    list_of_faces.append(roi_face)
		cv2.imwrite('faces_detected.pgm',self.img)
		return list_of_faces
		    

	def saveDetectedFaces(self,list_of_faces):
		'''
        This module saves the detected faces as images,separately,
        in the current directory.
        
        #Parameters: List of detected faces.
        #Return : None.
        
        '''
        self.number_of_detected_faces = len(list_of_faces)
		for i in range(self.number_of_detected_faces):
			cv2.imwrite('face_'+str(i+1)+'.pgm',list_of_faces[i])
