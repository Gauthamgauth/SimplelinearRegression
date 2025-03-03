#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_diabetes

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[2]:


X,y = load_diabetes(return_X_y=True)


# In[3]:


print(X.shape)
print(y.shape)


# In[4]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[6]:


reg = LinearRegression()
reg.fit(X_train,y_train)


# In[7]:


print(reg.coef_)
print(reg.intercept_)


# In[8]:


y_pred = reg.predict(X_test)
r2_score(y_test,y_pred)


# In[10]:


import random
class MGDRegression:
    def __init__(self,batch_size,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
    def fit(self,X_train,y_train):
        # init coef
        self.intercept_ = 0 
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            
            for j in range(int(X_train.shape[0]/self.batch_size)):
                
                idx = random.sample(range(X_train.shape[0]),self.batch_size)
            #updating coef and intercep_
            y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_
            print(y_hat.shape)
            intercept_der = -2 * np.mean(y_train[idx]-y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            
            coef_def = -2 *np.dot((y_train[idx] - y_hat),X_train[idx])
            self.coef_ = self.coef_ - (self.lr * self.coef_)
        
        
            print(self.intercept_,self.coef_)
        
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_


# In[29]:


mbr = MGDRegression(batch_size=int(X_train.shape[0]/10),learning_rate=0.10,epochs=50)


# In[30]:


mbr.fit(X_train,y_train)


# In[31]:


y_pred = mbr.predict(X_test)


# In[32]:


r2_score(y_test,y_pred)

