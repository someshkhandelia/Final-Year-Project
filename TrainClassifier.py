from Partitioner import Partitioner
from KNNClassifier import KNNClassifier
from TreeClassifier import TreeClassifier
from tensorflow.contrib import learn
from sklearn.metrics import accuracy_score
import numpy as np


#~~~~~~~~~~~~~~~~~~~~~~~~~ DATA TO BE FILLED BY USER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

base_dir_name = 'CambridgeSampleData'
class_name = 's'
total_classes = 40
total_sample = 10
img_extension = '.pgm'
training_percent = 0.8

#~~~~~~~~~~~~~~~~~~~~~~~~~ DATA TO BE FILLED BY USER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#




######## FEEDING DATA TO PARTITIONER #############################

partition_obj = Partitioner()
list_of_images = partition_obj.readData(base_dir_name,class_name,total_classes,total_sample,img_extension)
partition_obj.splitDataSet(training_percent)

######## GETTING DATA FROM PARTITIONER ##########################

train_data_x = partition_obj.getTrainingData()
train_target_y = partition_obj.getTrainingLabel()
test_data_x = partition_obj.getTestingData()
test_target_y = partition_obj.getTestingLabel()

####### KNN Classifier (Training and Testing) ###################

KNN_obj = KNNClassifier()
KNN_obj.trainClassifier(train_data_x,train_target_y)
KNN_acc_score = KNN_obj.testClassifier(test_data_x,test_target_y)
print('Accuracy of KNN classifier: ')
print(KNN_acc_score)
KNN_obj.saveClassifier('KNN_Classifier_'+ str(int(training_percent*100)) + '_' + base_dir_name + '_' + str(int(KNN_acc_score*100)) + '_percent_accurate')
print('############ KNN_CLASSIFIER SAVED SUCCESSFULLY !! ##############')

####### Tree Classifier (Training and Testing) ##################

Tree_obj = TreeClassifier()
Tree_obj.trainClassifier(train_data_x,train_target_y)
Tree_acc_score = Tree_obj.testClassifier(test_data_x,test_target_y)
print('Accuracy of Decision Tree classifier: ')
print(Tree_acc_score)
Tree_obj.saveClassifier('Tree_Classifier_'+ str(int(training_percent*100)) + '_' + base_dir_name + '_' + str(int(Tree_acc_score*100)) + '_percent_accurate')
print('############ DECISION_TREE_CLASSIFIER SAVED SUCCESSFULLY !! ##############')

##############################################################
