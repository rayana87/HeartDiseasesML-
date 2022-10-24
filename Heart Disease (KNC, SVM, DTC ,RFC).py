#!/usr/bin/env python
# coding: utf-8

# # Predicting Heart Diseases using Machine Learning
# Rayana ID#44286241 - Sarah ID#44286527 - Walaa ID#44285472

# ![image.png](attachment:image.png)

# Rayana ID#44286241

# ![image.png](attachment:image.png)

# Rayana ID#44286241

# ![image.png](attachment:image.png)

# Rayana ID#44286241

# ## Implementation of Heart diseases using Machine Learning techniques to predict whether a person is suffering from Heart Disease or not.
# 1. K Neighbors Classifier.
# 2. Support Vector Classifier.
# 3. Decision Tree Classifier.
# 4. Random Forest Classifier.
# 
# The used dataset is from this link --> (https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

# ## Importing the neccessary libraries

# In[2]:


# Rayana ID#44286241
import numpy as np # Linear algebra such as to work with arrays
import pandas as pd # to work with csv files & dataframes
import matplotlib.pyplot as plt # To create charts using pyplot
from matplotlib import rcParams # define the parameters
from matplotlib.cm import rainbow # color the parameters
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings # To ignore all warnings due to past/future depreciation of a feature
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('pip', 'install plotly')
import plotly
import plotly.graph_objs as go
import numpy as np   # So we can use random numbers in examples
import plotly.express as px


# Must enable in order to use plotly off-line (vs. in the cloud... hate cloud)
plotly.offline.init_notebook_mode()


# In[3]:


# Rayana ID#44286241
from sklearn.model_selection import train_test_split # To split the dataset into training and testing data
from sklearn.preprocessing import StandardScaler # to scale all the features with better adapts to the dataset


# ### Import all the Machine Learning algorithms.
# 1. K Neighbors Classifier
# 2. Support Vector Classifier
# 3. Decision Tree Classifier
# 4. Random Forest Classifier

# In[4]:


# Rayana ID#44286241
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[5]:


# Rayana ID#44286241
# Using the pandas read_csv method to read the dataset.
dataset = pd.read_csv('C:/Users/RAYANAH/Desktop/heart.csv') # this path should change according to your file location.

# after loaded the dataset, a quick look to the data
dataset.info()


# The the dataset has a total of 303 rows and there are no missing values. 
# 
# There are a total of 13 features along with one target value which we wish to find.
# 
# There are no missing values so we don’t need to take care of any null values. Next, I used describe() method.

# In[6]:


# Rayana ID#44286241
dataset.describe()


# The scale of each feature column is different and quite varied as well. While the maximum for age reaches 77 and the maximum of chol (serum cholestoral) is 564.
# Thus, feature scaling must be performed on the dataset.

# # Visualize the data
# To better understand our data and then look at any processing we might want to do.
# 
# ### Correlation Matrix
# To begin with, let’s see the correlation matrix of features and try to analyse it. The figure size is defined to 15 x 12 by using rcParams. Then, useing pyplot to show the correlation matrix. Using xticks and yticks, the names were added to the correlation matrix. colorbar() shows the colorbar for the matrix.

# In[7]:


#Sarah ID#44286527
rcParams['figure.figsize'] = 25, 15
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()


# In[8]:


# Walaa ID#44285472
import seaborn as sns

corr_dataset = dataset.corr() # Checks the correlation between numerical values in the data..
sns.clustermap(corr_dataset,annot= True,fmt = '.2f')
#While the annot shows the numerical values on the graph, fmt determines how many digits will be shown after the comma.
plt.title('Correlation Between Features')
plt.show();


# In[9]:


# Rayana ID#44286241
#Box up. needs a pre-melted process.
dataset_melted = pd.melt(dataset,id_vars='target',
                      var_name='Features',
                      value_name='Value')

plt.figure()
sns.boxplot(x='Features',y='Value',hue='target',data=dataset_melted) #Features are separated by target.
plt.xticks(rotation=75) #Feature names will be seen upright at 90 degrees.
plt.show()


# There is no single feature that has a very high correlation with our target value. Also, some of the features have a negative correlation with the target value and some have positive.

# ### Histogram
# A single command to draw the plots and it provides information for each variable by using --> dataset.hist().

# In[10]:


# Sarah ID#44286527
dataset.hist()


# The bars are discrete, which means that each is actually a categorical variable. We will need to handle these categorical variables before applying Machine Learning. Our target labels have two classes, 0 for no disease and 1 for disease.
# 
# We can observe form the histograms above, that each feature has a different range of distribution. using scaling before making the prediction is important.
# Making the target classes equal in size. 

# In[11]:


#Rayana ID#44286241
rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'Blue'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')

#For x-axis we used the unique() values from the target column and then set their name using xticks. 
#For y-axis, I used value_count() to get the values for each class. 
#I colored the bars as Blue and red.


# The two classes are balanced (not exactly 50%)but the ratio is good enough to continue without dropping or increasing the data.
# 
# Let’s say we have a dataset of 100 people with 99 non-patients and 1 patient. Without even training and learning anything, the model can always say that any new person would be a non-patient and have an accuracy of 99%. However, as we are more interested in identifying the 1 person who is a patient, we need balanced datasets so that our model actually learns.

# # Processing the Data
# Now, the categorical variables should be break each categorical column into dummy columns with 1s and 0s.
# Example, the column gender with values 1 for male & 0 for female, by converted into columns with value 1 true and 0 false.
# 
# 1. Using the get_dummies method to create dummy columns for categorical variables form pandas.
# 2. scaling the data by using StandardScaler -> fit_transform() method.

# In[12]:


#Rayana ID#44286241 - Sarah ID#44286527 - Walaa ID#44285472
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler() # StandardScaler from sklearn to scale the dataset.
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# # Training the models
# 4 algorithms and various parameters. 
# Comparing the final models. 
# 67% of the dataset used for training purposes and 33% for testing.
# 
# # Machine Learning
# 1. Importing the train_test_split to split our dataset into training and testing datasets. 
# 2. Then, importing all Machine Learning models to use it for train and test the data.

# In[13]:


#Rayana ID#44286241 - Sarah ID#44286527 - Walaa ID#44285472
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# # 1) K Neighbors Classifier
# This classifier looks for the classes of K nearest neighbors of a given data point and based on the majority class, it assigns a class to this data point. However, the number of neighbors can be varied. we varied them from 1 to 20 neighbors and calculated the test score in each case.
# 
# The classification score varies based on different values of neighbors that we choose. Thus,ploting the score graph for different values of K (neighbors) and check when do we achieve the best score.

# In[14]:


#Rayana ID#44286241
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))


# Now, the different neighbor values in the array knn_scores are scored. ploting it to see which value of K get the best scores.
# 
# The below is a line graph of the number of neighbors and the test score achieved in each case.

# In[15]:


#Rayana ID#44286241
rcParams['figure.figsize'] = 15, 9

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')

As you can see, the maximum score is 87% when the number of neighbors was chosen to be 8.
# # 2) Support Vector Classifier
# This classifier form a hyperplane that can separate the classes as much as possible by adjusting the distance between the data points and the hyperplane. There are several kernels based on which the hyperplane is decided. 
# 
# Four kernels are tested : linear, poly, rbf, and sigmoid --> checking which kernels has the best score.

# In[16]:


#Sarah ID#44286527
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(svc_classifier.score(X_test, y_test))


# In[17]:


# Sarah ID#44286527 
#ploting a bar plot of scores for each kernel and see which performed the best.

colors = rainbow(np.linspace(0, 1, len(kernels))) # rainbow method to select different colors for each bar and plot 
plt.bar(kernels, svc_scores, color = colors)
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier scores for different kernels')


# In[18]:


# Sarah ID#44286527
#The linear kernel had the best score and performed the best with 83%.
print("The score for Support Vector Classifier is {}% with {} kernel.".format(svc_scores[0]*100, 'linear'))


# # 3) Decision Tree Classifier
# This classifier creates a decision tree based on which, it assigns the class values to each data point. Here, we can vary the maximum number of features to be considered while creating the model. the features are in range from 1 to 30 (the total features in the dataset after dummy columns were added).
# 
# Here, varying between a set of max_features and see which returns the best accuracy.

# In[19]:


#Sarah ID#44286527 
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))


# In[20]:


# Sarah ID#44286527 
# Once we have the scores, we can then plot a line graph and see the effect of the number of features on the model scores.
#the maximum number of features is from 1 to 30 for split. 
#Now, let's see the scores for each of those cases.

rcParams['figure.figsize'] = 15, 9
plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')


# We can see form the above line graph, that the maximum score is 79% and is achieved for maximum features being selected to be either 2, 4 or 18.
# The model achieved the best accuracy at three values of maximum features, 2, 4 and 18.

# # 4) Random Forest Classifier
# 
# This classifier similar to Decision Trees. It creates a forest of trees where each tree is formed by a random selection of features from the total features. Here, we can used different number of trees to predict the class. The test scores are calculate with: 10, 100, 200, 500 and 1000 trees to see the effect.

# In[21]:


# Walaa ID#44285472
rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))

Then, plot the scores across a bar graph to see the best results. 
To avoid a continuous plot from 10 to 1000 because the continuous values are hard to decipher.
The X values are not setting as direct array, instead using the X values as [1, 2, 3, 4, 5]. Then,renamed them using xticks.
# In[22]:


# Walaa ID#44285472
# after training the model.a bar plot to compare the scores.

rcParams['figure.figsize'] = 10, 5
colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')


# In[23]:


# Walaa ID#44285472
print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[1]*100, [100, 500]))


# # Conclusion
# 
# Machine Learning techniques have been used in this project to predict if a person has a heart disease or not. Four models were trained and tested with scores: K Neighbors Classifier, Support Vector Classifier, Decision Tree Classifier and Random Forest Classifier.
# Different parameters are used across each model. Finally, the K Neighbors Classifier achieved the highest score with 87% with 8 nearest neighbors.

# In[24]:


# Walaa ID#44285472
print("K Neighbors Classifier score is {}% ".format(knn_scores[7]*100))

print("Support Vector Classifier score is {}% ".format(svc_scores[0]*100))

print("Decision Tree Classifier score is {}% ".format(dt_scores[1]*100))

print("Random Forest Classifier Score is {}% ".format(rf_scores[1]*100))

maxValue = max(knn_scores[7],svc_scores[0],dt_scores[1],rf_scores[2])
print( "K Neighbors Classifier has best score of {}%" .format(maxValue * 100))


# In[25]:


# Rayana ID#44286241 
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[26]:


# Rayana ID#44286241
scores=[knn_scores[7],svc_scores[0],dt_scores[1],rf_scores[1]]
AlgorthmsName=["K-NN","SVM","Decision Tree","Random Forest"]

#create traces

trace1 = go.Scatter(
    x = AlgorthmsName,
    y= scores,
    name='Algortms Name',
    marker =dict(color='rgba(0,255,0,0.5)',
               line =dict(color='rgb(0,0,0)',width=2)),
                text=AlgorthmsName
)
data = [trace1]

layout = go.Layout(barmode = "group",
                  xaxis= dict(title= 'ML Algorithms',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Prediction Scores',ticklen= 5,zeroline= False))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# ![image.png](attachment:image.png)

# In[ ]:




