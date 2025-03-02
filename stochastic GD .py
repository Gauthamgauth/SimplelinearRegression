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


# In[4]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[5]:


reg = LinearRegression()
reg.fit(X_train,y_train)


# In[6]:


print(reg.coef_)
print(reg.intercept_)


# In[7]:


y_pred = reg.predict(X_test)
r2_score(y_test,y_pred)


# In[28]:


class SGDRegression:
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
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0,X_train.shape[0])
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_
                
                intercept_der = -2 * (y_train[idx]-y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                coef_def = -2 *np.dot((y_train[idx] - y_hat),X_train[idx])
                self.coef_ = self.coef_ - (self.lr * self.coef_)
            
        
            print(self.intercept_,self.coef_)
        
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_


# In[29]:


coef_= np.ones(X_train.shape[1])
intercept_ = 0


# In[48]:


sgd = SGDRegression(learning_rate=0.01,epochs=1)


# In[49]:


sgd.fit(X_train,y_train)


# In[50]:


y_pred = sgd.predict(X_test)


# In[51]:


r2_score(y_test,y_pred)


# In[ ]:




