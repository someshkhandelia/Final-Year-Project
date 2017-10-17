from ImageCapture import ImageCapture
from FaceDetector import FaceDetector
from Preprocessor import Preprocessor
from ImageFeederKNN import ImageFeederKNN
from Partitioner import Partitioner
import cv2
from sklearn.metrics import accuracy_score



#~~~~~~~~~~~~~~~~~~~~~~~~~ DATA TO BE FILLED BY USER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

base_dir_name = 'CambridgeSampleData'
class_name = 's'
total_classes = 40
total_sample = 10
img_extension = '.pgm'
training_percent = 0.8
trained_pickle_name = 'KNN_Classifier_80_CambridgeSampleData_92_percent_accurate'

#~~~~~~~~~~~~~~~~~~~~~~~~~ DATA TO BE FILLED BY USER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#




######## FEEDING DATA TO PARTITIONER #############################

partition_obj = Partitioner()
list_of_images = partition_obj.readData(base_dir_name,class_name,total_classes,total_sample,img_extension)
partition_obj.splitDataSet(training_percent)

######## GETTING DATA FROM PARTITIONER ##########################

test_data_x = partition_obj.getTestingData()
test_target_y = partition_obj.getTestingLabel()

########################################################

obj4 = ImageFeederKNN()
obj4.convertRawDataToTestData(test_data_x)
predictions = obj4.getPrediction(trained_pickle_name)
print("Accuracy:")
print(accuracy_score(test_target_y,predictions))
print('Expectations:')
print(test_target_y)
print('Predictions:')
print(predictions)

