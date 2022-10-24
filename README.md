# HeartDiseasesML-
Implementation of Heart diseases using Machine Learning

Machine Learning is used across many spheres around the world including the healthcare industry, Machine Learning can play an essential role in predicting of Heart diseases and more. Such information can provide important insights to doctors who can then adapt their diagnosis and treatment per patient basis.Some application areas in which ML techniques are used includes Banking, Fraud detection, Bio-informatics, Marketing, Insurance, and Healthcare.

Our topic is about prediction of heart disease by processing patients' dataset to whom we need to predict the chance of occurrence of a heart disease. 

#### This notebook contains four algorithms of ML, as follow:  
* K Neighbors Classifier -->  looks for the classes of K nearest neighbors of a given data point and based on the majority class, it assigns a class to this data point.
* Support Vector Classifier --> form a hyperplane that can separate the classes as much as possible by adjusting the distance between the data points
* Decision Tree Classifier --> creates a decision tree, it assigns the class values to each data point.
* Random Forest Classifier --> Similar to Decision Trees. It creates a forest of trees where each tree is formed by a random selection of features from the total features

#### The Dataset
The dataset has a total of 303 rows and there are no missing values. There are a total of 13 features along with one target value which we wish to find.
There are no missing values so we don’t need to take care of any null values.

The Attributes are (sex), (age), (chest pain), (blood pressure), (blood sugar), (ECG), (heart rate) and (disease).

Attribute age is classified in three groups. 
1-young, 2 - Middle age 3 - old age.

Attribute chest pain is categorized in four groups.
1- Asymptomatic 2- Angina Pectoris, 3- non-angian, 4-typ-angian

Attribute blood pressure is classified in five groups.
1- Normal, 2-elevated, 3-HBP Stage1, 4-HBP Stage2, 5- Hypertension Crisis.

Attribute blood sugar is classified in 3 groups.
1-Normal, 2-pre diabetic, 3-diabetic

Attribute ECG is classified in 3 groups.
1- left ventricular hypertrophy, 2-normal, 3-st-t-aveabnormality

Attribute heart rate is classified in 3 groups.
1- Good, 2-average, 3-poor

Attribute disease is classified in 2 groups.
1 negative and 2-positive.


A scaling technique used before feeding in to the mL models, because there are difference in ranges of features will cause different step sizes for each feature. Distance algorithms like KNN, K-means, and SVM are most affected by the range of features. This is because behind the scenes they are using distances between data points to determine their similarity. This will impact the performance of the machine learning algorithms.

