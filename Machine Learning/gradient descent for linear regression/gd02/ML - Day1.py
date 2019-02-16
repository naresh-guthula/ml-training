#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
myData = pandas.read_csv('rentals.csv')


# In[2]:


print (myData)


# In[3]:


print (myData.shape)


# In[4]:


type(myData)


# In[5]:


myData.head()


# In[6]:


myData.head(10)


# In[7]:


myData.tail()


# In[8]:


myData.tail(8)


# In[9]:


myData.describe()


# In[10]:


from pandas.plotting import scatter_matrix
myData.plot(kind='box',subplots=True,layout=
(2,2),sharex=False,sharey=False)

scatter_matrix(myData)

import matplotlib.pyplot as plt
plt.show()


# In[11]:


dataX = pandas.DataFrame({'area': myData.area})


# In[12]:


dataX


# In[13]:


dataY = pandas.DataFrame({'cost': myData['cost']})
dataY


# In[14]:


from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(dataX, dataY,                                                test_size = 0.20, random_state = 11)


# In[15]:


trainX.head()


# In[16]:


trainY.head()


# In[17]:


trainX.shape


# In[18]:


testX.head()


# In[45]:


from sklearn.linear_model import SGDRegressor
model = SGDRegressor(shuffle = False, eta0 = .0000001, max_iter =100, tol = 1000)


# In[46]:


model.fit(trainX, trainY.values.ravel())


# In[47]:


print ('Coefficients: ', model.coef_)
print ('Intercept: ', model.intercept_)

print('iterations ran :',model.n_iter_);

print ('R2: ', model.score (testX, testY))


# In[22]:


ywhat = model.predict(testX)
ywhat


# In[ ]:




