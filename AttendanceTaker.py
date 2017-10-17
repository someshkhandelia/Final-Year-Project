from ImageCapture import ImageCapture
from FaceDetector import FaceDetector
from Preprocessor import Preprocessor
from ImageFeederKNN import ImageFeederKNN
from FileWriter import FileWriter
import cv2
import sys


### USAGE ---> python AttendanceTaker.py <course_code> 
### replace <course_code> by CS-403 etc. 



#~~~~~~~~~~~~~~~~~~~~~~~~~ DATA TO BE FILLED BY USER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

trained_pickle_name = '' ## give pickle to load here(without extension) ##

#~~~~~~~~~~~~~~~~~~~~~~~~~ DATA TO BE FILLED BY USER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#




######### capturing the image using webcam ##################

cam = cv2.VideoCapture(0)
ic_obj = ImageCapture()
img = ic_obj.getImage(cam)
ic_obj.saveCapturedImage(img)

######### Preprocessing the image ##########################

prp_obj = Preprocessor()
img = prp_obj.adjustContrast(img)

######## Detecting faces in the captured image #############

fd_obj = FaceDetector(img)
list_of_faces = fd_obj.detectFacesInImage()
#fd_obj.saveDetectedFaces(list_of_faces)

############################################################

print("############## FACE DETECTION COMPLETED !! ##################")
cam.release()
cv2.destroyAllWindows()

######## Using trained Classifier ##########################

ifk_obj = ImageFeederKNN()
ifk_obj.convertRawDataToTestData(list_of_faces)
predictions = ifk_obj.getPrediction(trained_pickle_name)
#print('The following roll numbers are present:')
#print(predictions)

####### FILE WRITER #################################

#predictions = [33,1,105,67]
try:
	course_code = str(sys.argv[1])
	course_code = course_code.upper()
	fw_obj = FileWriter(course_code)
	fw_obj.saveToFile(predictions)
except IndexError:
	print("Specify course_code \n### USAGE ---> python AttendanceTaker.py <course_code> \n### replace <course_code> by CS-403 etc.")







