#!/usr/bin/env python
# coding: utf-8

# <a id = 'intro'></a>
# # Boston data analysis using neural network

# Here, we will develop, evaluate and compare the performance and predictive power of a simple linear regression model and neural network on boston housing price data
# 
# Here the target is to determine the price of the property based on the features.

# <a id = 'data'></a>
# # Dataset

# The dataset used in the anlysis comes from the UCI Machine Learning Repository. This data was collected in 1978 and each of the 506 entries represents aggregate information about 14 features of homes from various suburbs located in Boston.
# 
# The features can be summarized as follows:
# * CRIM: This is the per capita crime rate by town
# * ZN: This is the proportion of residential land zoned for lots larger than 25,000 sq.ft.
# * INDUS: This is the proportion of non-retail business acres per town.
# * CHAS: This is the Charles River dummy variable (this is equal to 1 if tract bounds river; 0 otherwise)
# * NOX: This is the nitric oxides concentration (parts per 10 million)
# * RM: This is the average number of rooms per dwelling
# * AGE: This is the proportion of owner-occupied units built prior to 1940
# * DIS: This is the weighted distances to five Boston employment centers
# * RAD: This is the index of accessibility to radial highways
# * TAX: This is the full-value property-tax rate per 1000 bucks
# * PTRATIO: This is the pupil-teacher ratio by town
# * B: This is calculated as 1000(Bk — 0.63)², where Bk is the proportion of people of African American descent by town
# * LSTAT: This is the percentage lower status of the population
# * MEDV: This is the median value of owner-occupied homes in 1000s

# <a id = 'library'></a>
# # Importing libraries and the dataset
# Unlike the conventional way, I import the library when it is needed. It will actually help you to understand where the application of the class and it's function is used

# In[6]:


#Importing the pandas for data processing and numpy for numerical computing
import numpy as np
import pandas as pd


# In[ ]:





# In[7]:


# Importing the Boston Housing dataset from the sklearn
from sklearn.datasets import load_boston
boston = load_boston()


# In[8]:


#Converting the data into pandas dataframe
data = pd.DataFrame(boston.data)


# ### Look at the dataset

# In[10]:


#Adding the feature names to the dataframe
data.columns = boston.feature_names


# In[11]:


#Adding the target variable to the dataset
data['PRICE'] = boston.target 


# In[12]:


#Looking at the data with names and target variable
data.head()


# In[13]:


# Length of the data
print(data.shape)


# In[14]:


#Checking the missing values in the dataset
data.isnull().sum()


# <a id = 'visual'></a>
# # Visualisation

# In[17]:


#checking the distribution of the target variable
import seaborn as sns
sns.distplot(data.PRICE)


# The distribution seems normal, has not be the data normal we would have perform log transformation or took to square root of the data to make the data normal. Normal distribution is need for the machine learning for better predictiblity of the model

# <a id = 'split'></a>
# ### Splitting the dependent feature and independent feature 

# In[22]:


#X = data[['LSTAT','RM','PTRATIO']]
X = data.iloc[:,:-1]
y= data.PRICE


# <a id = 'valid'></a>
# ### Splitting the data for Model Validation 

# In[23]:


# Splitting the data into train and test for building the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)


# <a id = 'build'></a>
# ### Building the Model 

# In[24]:


#Linear Regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[25]:


#Fitting the model
regressor.fit(X_train,y_train)


# <a id = 'evaluate'></a>
# ### Model Evaluation

# In[26]:


#Prediction on the test dataset
y_pred = regressor.predict(X_test)


# In[27]:


# Predicting RMSE the Test set results
from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print(rmse)


# In[28]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)


# <a id  = 'NN'></a>
# ## Neural Networks

# In[29]:


#Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#  We are using Keras for developing the neural network.
# * Models in Keras are defined as a sequence of layers
# * We create a Sequential model and add layers one at a time with activation function
# * Activation function decides, whether a neuron should be activated or not by calculating weighted sum and further adding bias with it. The purpose of the activation function is to introduce non-linearity into the output of a neuron.The activation we are using is relu
# * As this is a regression problem, the output layer has no activation function
# * Elements of neural network has input layer, hidden layer and output layer
# * Model Compilation:- The compilation is the final step in creating a model. Once the compilation is done, we can move on to training phase.
# * Optimizer: - (Backpropagation) The optimizer we are using is adam. "Adam" is an optimization algorithm that can be used instead of the classical stochastic gradient descent (SGD) procedure to update network weights iterative based in training data.
# * Loss - mean square error

# In[30]:


#Creating the neural network model
import keras
from keras.layers import Dense, Activation,Dropout
from keras.models import Sequential

model = Sequential()

model.add(Dense(128,activation  = 'relu',input_dim =13))
model.add(Dense(64,activation  = 'relu'))
model.add(Dense(32,activation  = 'relu'))
model.add(Dense(16,activation  = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')


# In[31]:


model.fit(X_train, y_train, epochs = 100)


# <a id = 'eval'></a>
# ### Evaluation of the model

# In[32]:


y_pred = model.predict(X_test)


# In[33]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)


# In[34]:


# Predicting RMSE the Test set results
from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print(rmse)


# <a id = 'conclude'></a>
# ## Conclusion
# 
# Using a simple neural network, we were able to improve the model significantly.
