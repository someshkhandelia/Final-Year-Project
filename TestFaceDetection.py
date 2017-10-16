from ImageCapture import ImageCapture
from FaceDetector import FaceDetector
from Preprocessor import Preprocessor
from ImageFeederKNN import ImageFeederKNN
import cv2

cam = cv2.VideoCapture(0)

######### capturing the image using webcam ##################

obj1 = ImageCapture()
img = obj1.getImage(cam)
obj1.saveCapturedImage(img)

######### preferably do some preprocessing here ##############

obj2 = Preprocessor()
#cv2.imwrite('bright_img.jpg',obj2.increaseBrightness(img))
#cv2.imwrite('dark_img.jpg',obj2.decreaseBrightness(img))
#cv2.imwrite('equalised_img.jpg',obj2.adjustContrast(img))

img = obj2.adjustContrast(img)
######## Detecting faces in the captured image #############

obj3 = FaceDetector(img)
list_of_faces = obj3.detectFacesInImage()
obj3.saveDetectedFaces(list_of_faces)

####### Predicting using classifier ########################
'''
trained_pickle_name = ''
obj4 = ImageFeederKNN()
obj4.convertRawDataToTestData(list_of_faces)
predictions = obj4.getPrediction(trained_pickle_name)
print('Predictions:')
print(predictions)
'''
############################################################

cam.release()
cv2.destroyAllWindows()

