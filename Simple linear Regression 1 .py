#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np

class MyLR:
    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self, X_train, y_train):
        num = 0
        den = 0
        X_mean = X_train.mean()
        y_mean = y_train.mean()

        for i in range(X_train.shape[0]):  # Correct indentation
            num += (X_train[i] - X_mean) * (y_train[i] - y_mean)
            den += (X_train[i] - X_mean) ** 2  # More readable
        
        self.m = num / den
        self.b = y_mean - (self.m * X_mean)

        print("Slope (m):", self.m)
        print("Intercept (b):", self.b)

    def predict(self, X_test):
        return self.m * X_test + self.b


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv("archive (3).zip")


# In[4]:


df.head()


# In[5]:


X = df.iloc[:,0].values
y = df.iloc[:,1].values


# In[7]:


X


# In[8]:


y


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state=2)


# In[12]:


X_train.shape


# In[27]:


lr = MyLR()


# In[28]:


lr.fit(X_train,y_train)


# In[29]:


X_test[0]


# In[30]:


lr.predict(X_test[0])

