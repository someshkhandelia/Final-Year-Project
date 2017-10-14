from ImageCapture import ImageCapture
from FaceDetector import FaceDetector
from Preprocessor import Preprocessor
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


cam.release()
cv2.destroyAllWindows()

