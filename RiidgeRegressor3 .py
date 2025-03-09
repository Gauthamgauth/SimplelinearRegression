#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np 


# In[2]:


X,y = load_diabetes(return_X_y=True)


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)


# In[5]:


from sklearn.linear_model import SGDRegressor


# In[13]:


reg = SGDRegressor(penalty="l2",max_iter=500,eta0=0.1,learning_rate="constant",alpha=0.001)


# In[14]:


reg.fit(X_train,y_train)


# In[20]:


reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("r2 score",r2_score(y_test,y_pred))
print(reg.coef_)
print(reg.intercept_)


# In[21]:


from sklearn.linear_model import Ridge


# In[22]:


reg = Ridge(alpha=0.001,max_iter=500,solver="sparse_cg")


# In[24]:


reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("r2 score",r2_score(y_test,y_pred))
print(reg.coef_)
print(reg.intercept_)


# In[41]:


class MyRidge:
    def __init__(self,epochs,learning_rate,alpha):
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        
        
    def fit(self,X_train,y_train):
        
        self.coef_ = np.ones(X_train.shape[1])
        self.intercept_ = 0
        thetha = np.insert(self.coef_,0,self.intercept_)
        
        X_train = np.insert(X_train,0,1,axis=1)
        
        for i in range(self.epochs):
            thetha_der = np.dot(X_train.T,X_train).dot(thetha) - np.dot(X_train.T,y_train) - self.alpha*thetha
            thetha = thetha - self.learning_rate*thetha_der
            
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_


# In[42]:


reg = MyRidge(epochs=500,alpha=0.001,learning_rate=0.005)


# In[43]:


reg.fit(X_train,y_train)


# In[44]:


y_pred = reg.predict(X_test)
print("r2 score",r2_score(y_test,y_pred))
print(reg.coef_)
print(reg.intercept_)

