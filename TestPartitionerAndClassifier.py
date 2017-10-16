from Partitioner import Partitioner
from NeuralNetwork import NeuralNetwork
from KNNClassifier import KNNClassifier
from TreeClassifier import TreeClassifier
from tensorflow.contrib import learn
from sklearn.metrics import accuracy_score
import numpy as np


partition_obj = Partitioner()
list_of_images = partition_obj.readData('CambridgeSampleData','s',40,10,'.pgm')

partition_obj.splitDataSet(0.8)

train_data_x = partition_obj.getTrainingData()
train_target_y = partition_obj.getTrainingLabel()
test_data_x = partition_obj.getTestingData()
test_target_y = partition_obj.getTestingLabel()


############# CONVERSION TO NP ARRAY ################################

train_data_x = np.asarray(train_data_x)
train_target_y = np.asarray(train_target_y)
test_data_x = np.asarray(test_data_x)
test_target_y = np.asarray(test_target_y)


####### KNN Classifier #########################################

KNN_obj = KNNClassifier()
KNN_obj.trainClassifier(train_data_x,train_target_y)
KNN_acc_score = KNN_obj.testClassifier(test_data_x,test_target_y)
print('Accuracy of KNN classifier: ')
print(KNN_acc_score)
KNN_obj.saveClassifier('KNN_Classifier_80_Cambridge')
print("Expected: ")
print(test_target_y)
print("Predicted: ")
print(KNN_obj.predicted_results)

####### Tree Classifier ########################################

Tree_obj = TreeClassifier()
Tree_obj.trainClassifier(train_data_x,train_target_y)
Tree_acc_score = Tree_obj.testClassifier(test_data_x,test_target_y)
print('Accuracy of Tree classifier: ')
print(Tree_acc_score)
Tree_obj.saveClassifier('Tree_Classifier_80_Cambridge')
print("Expected: ")
print(test_target_y)
print("Predicted: ")
print(Tree_obj.predicted_results)


