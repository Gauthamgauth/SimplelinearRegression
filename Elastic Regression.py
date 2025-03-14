#!/usr/bin/env python
# coding: utf-8

# In[17]:


from sklearn.datasets import load_diabetes 
from sklearn.linear_model import ElasticNet,LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score


# In[9]:


X,y = load_diabetes(return_X_y=True)


# In[10]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[11]:


# linearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
r2_score(y_pred,y_test)


# In[15]:


# ridge 
reg = Ridge(alpha=0.1)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
r2_score(y_pred,y_test)


# In[18]:


# elastic 
reg = Lasso(alpha=0.01)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
r2_score(y_pred,y_test)


# In[20]:


# elastic 
reg = ElasticNet(alpha=0.005,l1_ratio=0.9)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
r2_score(y_test,y_pred)

