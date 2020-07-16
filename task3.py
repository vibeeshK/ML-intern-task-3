#!/usr/bin/env python
# coding: utf-8

# In[15]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
iris=datasets.load_iris()


# In[16]:


# Forming  dataframe
data=pd.DataFrame(iris.data, columns=iris.feature_names)
data['target']=iris.target
data


# In[17]:


#Split training and testing data

y=df['target']
x=df.drop('target',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)


# In[20]:


#DecisionTreeClassifier
dt_model=DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
dt_pred = dt_model.predict(x_test)

msee1 = mean_squared_error(y_test, dt_pred)
ra1 = r2_score(y_test, dt_pred)
maee1 = mean_absolute_error(y_test,dt_pred)
print("Mean Squared Error:",msee1)
print("R score:",ra1)
print("Mean Absolute Error:",maee1)
print('accuracy score:')
print(accuracy_score(y_test,dt_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:





# In[ ]:





# In[ ]:




