import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cam = cv2.VideoCapture(1)

cou = 0
while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    cou = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_face = img[y:y+h,x:x+w]
        cou = cou+1
        cv2.imwrite('face_'+str(cou)+'.jpg',roi_face)

    #print (len(faces))
    
    
    cv2.imshow('img',img)
    k = cv2.waitKey(50) & 0xff

    if k==27:
        break
cam.release()
cv2.destroyAllWindows()
