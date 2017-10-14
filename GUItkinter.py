from ImageCapture import ImageCapture
from FaceDetector import FaceDetector
import cv2

cam = cv2.VideoCapture(0)
obj1 = ImageCapture()
img = obj1.getImage(cam)
obj1.saveCapturedImage(img)

######### preferably do some preprocessing here ##############

obj2 = FaceDetector(img)
list_of_faces = obj2.detectFacesInImage()
obj2.saveDetectedFaces(list_of_faces)
cam.release()
cv2.destroyAllWindows()

