# Attendance Automator
Hi! I am Somesh Khandelia, a senior at National Institute of Technology, Durgapur, India.
This is my final year project, as an undergraduate, under the guidance of Dr. Bibhash Sen.
As the name suggests, this project automates the attendance taking process.

## What is basically happening?
To put it as simply as possible, Face Detection is being used to identify the students in a class.
Their attendance is automatically recorded in a separate file arranged according to **Course code** and **Date**.

## What do you need to use this?
1. Python 3.5.x
2. Python libraries:
    * OpenCv
    * Numpy
    * Scipy
    * Scikit-learn
    
Obviously, you need the **training data** to train the classifier to identify the students.
Training data comprises of a good number of **images** of each student. The images should satisfy the following:
1. Equal number of images for each student.
2. Images should be of small size.
3. Images should be uniformly taken.

## How to use?
The usability is extremely straightforward and lucid.
### Provide the training data
The data for training the classifier should be provided following certain conventions.
Let us assume a few things, and it will be simpler to understand with an example. <br />
> * Suppose we have **50** students in a class. We are keeping all the images in a folder called **Images_for_training**. <br />
> * Inside that folder we will have **50** directories, each containing say **20** images of a particular student respectively. <br />
> * All of these 50 directories will have a common prefix in their name say **Roll** followed by the roll number(in digits only). <br />
> * Therefore a directory maybe called **Roll45**. <br />
> * Inside each such directory 20 images are present named '1.pgm', '2.pgm' and so on. <br />
> (Assuming **.pgm** is the extension we are using for our images)
### Train the Classifier
You need to modify certain variables in **TrainClassifier.py** <br />
> * base_dir_name = 'Images_for_training'
> * class_name = 'Roll'
> * total_classes = 50
> * total_sample = 20
> * img_extension = '.pgm'
> * training_percent = 0.8 <br />

**training_percent** signifies what percent of the total data you are using to train the classifier. <br />
The remaining will be used to test the classifier and tell you how accurate the classifier is. <br />

Let us now run **TrainClassifier.py**. In a terminal, execute:
> python TrainClassifier.py

### Run the script!
You will obtain two **pickle** (.pkl) files.
* **KNN Classifier**'s pickle file <br />
* **Decision Tree Classifier**'s pickle file <br />

Suppose we decide to work with KNN Classifier's pkl file. <br />
Suppose the name of this file is **KNN_Classifier_something.pkl** <br />

Now we need to modify a variable in **AttendanceTaker.py**
> * trained_pickle_name = 'KNN_Classifier_something' <br />

We are all set to run our script! <br />
Suppose the **Course code** is **CS-403**
Execute the following on a terminal
> * python AttendanceTaker.py CS-403 <br />

And that's it !! <br />
Attendance gets stored in a folder called **Attendance**.


