import cv2
import numpy as np


class ImageCapture:
    '''
    This class is for capturing image from a webcam.
    '''

    def __init__(self):
        pass

    
    def captureFrame(self,cam):
        '''
        This module captures a single frame and returns it.
        
        #Parameters: VideoCapture object from openCv library.
        #Return : Frame captured.
        
        '''
        retval, frame = cam.read()
        return frame

    def getImage(self,cam):
        '''
        This module gets an image from captureFrame().
        It takes into account the redundant frames that are clicked,
        when the camera is trying to adjust to the surroundings.
        
        #Parameters: VideoCapture object from openCv library.
        #Return : Image captured.
        
        '''
        redundant_frames = 30
        print("Taking image...")
        for i in range(redundant_frames):
            temp = self.captureFrame(cam)
        img = self.captureFrame(cam)
        return img

    def saveCapturedImage(self,img):
        '''
        This module saves the captured image,
        in the current directory.
        
        #Parameters: Captured image.
        #Return : None.

        '''
        cv2.imwrite('captured_image.jpg',img)




