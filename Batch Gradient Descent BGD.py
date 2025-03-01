#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_diabetes

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[3]:


X,y = load_diabetes(return_X_y=True)


# In[5]:


print(X.shape)
print(y.shape)


# In[7]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[8]:


reg = LinearRegression()
reg.fit(X_train,y_train)


# In[9]:


print(reg.coef_)
print(reg.intercept_)


# In[10]:


y_pred = reg.predict(X_test)
r2_score(y_test,y_pred)


# In[30]:


class GDRegression:
    def __init__(self,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X_train,y_train):
        # init coef
        self.intercept_ = 0 
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range (self.epochs):
            #updating coef and intercep_
            y_hat = np.dot(X_train,self.coef_) + self.intercept_
            print(y_hat.shape)
            intercept_der = -2 * np.mean(y_train-y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            
            coef_def = -2 *np.dot((y_train - y_hat),X_train)/X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * self.coef_)
        
        
            print(self.intercept_,self.coef_)
        
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_


# In[31]:


gdr = GDRegression(epochs=1)


# In[32]:


gdr.fit(X_train,y_train)


# In[33]:


gdr = GDRegression(epochs=100)


# In[34]:


gdr.fit(X_train,y_train)


# In[43]:


gdr = GDRegression(epochs=1000,learning_rate=0.4)


# In[44]:


r2_score(y_pred,y_test)

