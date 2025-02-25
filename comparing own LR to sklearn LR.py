#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_diabetes 


# In[2]:


X,y = load_diabetes(return_X_y=True)


# In[3]:


X


# In[4]:


X.shape


# In[5]:


y.shape


# In[6]:


from sklearn.linear_model import LinearRegression


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)


# In[15]:


print(X_train.shape)
print(X_test.shape)


# In[17]:


reg = LinearRegression()


# In[19]:


reg.fit(X_train,y_train)


# In[20]:


y_pred = reg.predict(X_test)


# In[21]:


from sklearn.metrics import r2_score


# In[22]:


r2_score(y_test,y_pred)


# In[93]:


reg.coef_


# In[94]:


reg.intercept_


# In[27]:


#OWN LINEAR REGRESSION 


# In[143]:


import numpy as np

class MyLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None  

    def fit(self, X_train, y_train): 
        X_train = np.insert(X_train, 0, 1, axis=1)  
        betas = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        
        self.intercept_ = betas[0] 
        self.coef_ = betas[1:]      

    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)  
        y_pred = np.dot(X_test, np.r_[self.intercept_, self.coef_])  
        return y_pred


# In[144]:


lr = MyLR()


# In[145]:


lr.fit(X_train,y_train)


# In[146]:


X_train.shape


# In[147]:


np.insert(X_train,0,1,axis = 1)


# In[148]:


np.insert(X_train,0,1,axis = 1).shape


# In[149]:


lr.predict(X_test)


# In[150]:


y_pred = lr.predict(X_test)


# In[151]:


r2_score(y_test,y_pred)


# In[152]:


lr.coef_


# In[153]:


lr.intercept_

