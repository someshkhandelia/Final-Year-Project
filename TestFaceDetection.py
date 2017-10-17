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

'''im1 = cv2.imread('captured_image.pgm',-1)
cv2.imshow('captured_pgm',im1)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

######### preferably do some preprocessing here ##############

obj2 = Preprocessor()
img = obj2.adjustContrast(img)

######## Detecting faces in the captured image #############

obj3 = FaceDetector(img)
list_of_faces = obj3.detectFacesInImage()
obj3.saveDetectedFaces(list_of_faces)

############################################################

cam.release()
cv2.destroyAllWindows()

