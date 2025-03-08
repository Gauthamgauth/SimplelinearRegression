#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


X,y = make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=20,random_state=13)


# In[4]:


plt.scatter(X,y)


# In[5]:


from sklearn.linear_model import LinearRegression


# In[7]:


lr = LinearRegression()
lr.fit(X,y)
print(lr.coef_)
print(lr.intercept_)


# In[8]:


from sklearn.linear_model import Ridge


# In[11]:


rr = Ridge(alpha=10)
rr.fit(X,y)
print(rr.coef_)
print(rr.intercept_)


# In[12]:


rr1 = Ridge(alpha=100)
rr1.fit(X,y)
print(rr1.coef_)
print(rr1.intercept_)


# In[13]:


plt.plot(X,y,"b.")
plt.plot(X,lr.predict(X),color="red",label="alpha-0")
plt.plot(X,rr.predict(X),color="green",label="alpha-0")
plt.plot(X,rr1.predict(X),color="orange",label="alpha-0")


# In[23]:


class MyRidge:
    def __init__(self,alpha=0.1):
        self.alpha = alpha
        self.m = None
        self.b = None
        
    def fit(self,X_train,y_train):
        num = 0
        den = 0 
        
        for i in range(X_train.shape[0]):
            num = num+(y_train[i] - y_train.mean())*(X_train[i]-X_train.mean())
            den = den+(X_train[i] - y_train.mean())*(X_train[i]-X_train.mean()) + self.alpha

        
        self.m = num/(den + self.alpha)
        self.b = y_train.mean() - (self.m*X_train.mean())
        print(self.m , self.b)
        
    def predict(X_test):
        pass
        


# In[24]:


reg = MyRidge(alpha=10)


# In[25]:


reg.fit(X,y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




