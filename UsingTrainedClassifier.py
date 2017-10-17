from ImageCapture import ImageCapture
from FaceDetector import FaceDetector
from Preprocessor import Preprocessor
from ImageFeederKNN import ImageFeederKNN
from Partitioner import Partitioner
import cv2
from sklearn.metrics import accuracy_score

########## GETTING LIST OF FACES(Cambridge only) #########

partition_obj = Partitioner()
list_of_images = partition_obj.readData('CambridgeSampleData','s',40,10,'.pgm')
training_percent = 0.8
partition_obj.splitDataSet(training_percent)
test_data_x = partition_obj.getTestingData()
test_target_y = partition_obj.getTestingLabel()

########################################################
trained_pickle_name = 'KNN_Classifier_80_Cambridge_92_percent_accurate'
obj4 = ImageFeederKNN()
obj4.convertRawDataToTestData(test_data_x)
predictions = obj4.getPrediction(trained_pickle_name)
print("Accuracy:")
print(accuracy_score(test_target_y,predictions))
print('Expectations:')
print(test_target_y)
print('Predictions:')
print(predictions)

