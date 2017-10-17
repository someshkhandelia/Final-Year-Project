from ImageCapture import ImageCapture
from FaceDetector import FaceDetector
from Preprocessor import Preprocessor
from ImageFeederKNN import ImageFeederKNN
import cv2


#~~~~~~~~~~~~~~~~~~~~~~~~~ DATA TO BE FILLED BY USER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

trained_pickle_name = '' ## give pickle to load here(without extension) ##

#~~~~~~~~~~~~~~~~~~~~~~~~~ DATA TO BE FILLED BY USER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#




######### capturing the image using webcam ##################

cam = cv2.VideoCapture(0)
obj1 = ImageCapture()
img = obj1.getImage(cam)
obj1.saveCapturedImage(img)

######### Preprocessing the image ##########################

obj2 = Preprocessor()
img = obj2.adjustContrast(img)

######## Detecting faces in the captured image #############

obj3 = FaceDetector(img)
list_of_faces = obj3.detectFacesInImage()
#obj3.saveDetectedFaces(list_of_faces)

############################################################

print("############## FACE DETECTION COMPLETED !! ##################")
cam.release()
cv2.destroyAllWindows()

######## Using trained Classifier ##########################

obj4 = ImageFeederKNN()
obj4.convertRawDataToTestData(list_of_faces)
predictions = obj4.getPrediction(trained_pickle_name)
print('The following roll numbers are present:')
print(predictions)



