from Partitioner import Partitioner
from NeuralNetwork import NeuralNetwork
from KNNClassifier import KNNClassifier
from TreeClassifier import TreeClassifier
from tensorflow.contrib import learn
from sklearn.metrics import accuracy_score


partition_obj = Partitioner()
list_of_images = partition_obj.readData('CambridgeSampleData','s',40,10,'.pgm')

partition_obj.splitDataSet(0.8)

train_data_x = partition_obj.getTrainingData()
train_target_y = partition_obj.getTrainingLabel()
test_data_x = partition_obj.getTestingData()
test_target_y = partition_obj.getTestingLabel()

########  NEURAL NETWORK ######################################

NN_obj = NeuralNetwork()
NN_obj.trainClassifier(train_data_x,train_target_y)
#NN_acc_score = NN_obj.testClassifier(test_data_x,test_target_y)
#print('Accuracy of DNN classifier: ')
#print(NN_acc_score)
NN_obj.testClassifier(test_data_x,test_target_y)
NN_obj.saveClassifier('DNN_Classifier_80_Cambridge')

####### KNN Classifier #########################################
'''
KNN_obj = KNNClassifier()
KNN_obj.trainClassifier(train_data_x,train_target_y)
KNN_acc_score = KNN_obj.testClassifier(test_data_x,test_target_y)
print('Accuracy of KNN classifier: ')
print(KNN_acc_score)
KNN_obj.saveClassifier('KNN_Classifier_80_Cambridge')
'''
####### Tree Classifier ########################################
'''
Tree_obj = TreeClassifier()
Tree_obj.trainClassifier(train_data_x,train_target_y)
Tree_acc_score = Tree_obj.testClassifier(test_data_x,test_target_y)
print('Accuracy of DNN classifier: ')
print(Tree_acc_score)
Tree_obj.saveClassifier('Tree_Classifier_80_Cambridge')
'''
