from Partitioner import Partitioner
from KNNClassifier import KNNClassifier
from TreeClassifier import TreeClassifier
from tensorflow.contrib import learn
from sklearn.metrics import accuracy_score
import numpy as np

######## FEEDING DATA TO PARTITIONER #############################

partition_obj = Partitioner()
list_of_images = partition_obj.readData('CambridgeSampleData','s',40,10,'.pgm')
training_percent = 0.8
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
KNN_obj.saveClassifier('KNN_Classifier_'+ str(int(training_percent*100)) + '_Cambridge_' + str(int(KNN_acc_score*100)) + '_percent_accurate')
print("Expected: ")
print(test_target_y)
print("Predicted: ")
print(KNN_obj.predicted_results)

####### Tree Classifier (Training and Testing) ##################

Tree_obj = TreeClassifier()
Tree_obj.trainClassifier(train_data_x,train_target_y)
Tree_acc_score = Tree_obj.testClassifier(test_data_x,test_target_y)
print('Accuracy of Tree classifier: ')
print(Tree_acc_score)
Tree_obj.saveClassifier('Tree_Classifier_'+ str(int(training_percent*100)) + '_Cambridge_' + str(int(Tree_acc_score*100)) + '_percent_accurate')
print("Expected: ")
print(test_target_y)
print("Predicted: ")
print(Tree_obj.predicted_results)


