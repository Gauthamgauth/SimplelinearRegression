#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


from sklearn.datasets import load_diabetes
data=load_diabetes()


# In[4]:


print(data.DESCR)


# In[5]:


X = data.data
y=data.target


# In[6]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


L = LinearRegression()


# In[9]:


L.fit(X_train,y_train)


# In[12]:


y_pred = L.predict(X_test)


# In[13]:


from sklearn.metrics import r2_score,mean_squared_error

print("r2_score",r2_score(y_test,y_pred))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))


# In[14]:


from sklearn.linear_model import Ridge
R=Ridge(alpha=0.0001)


# In[15]:


R.fit(X_train,y_train)


# In[16]:


y_pred1 = R.predict(X_test)


# In[17]:


from sklearn.metrics import r2_score,mean_squared_error

print("r2_score",r2_score(y_test,y_pred))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))


# In[19]:


m = 100
x1 = 5*np.random.rand(m,1)-2
x2 = 0.7 * x1 ** 2 -2 * x1 + 3 + np.random.randn(m,1)

plt.scatter(x1,x2)
plt.show()


# In[24]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_preds_ridge(x1,x2,alpha):
    model = Pipeline([
        ("poly_feats",PolynomialFeatures(degree=16)),
        ("rigde",Ridge(alpha=alpha))
        
    ])
    model.fit(x1,x2)
    return model.predict(x1)

alphas = [0,20,200]
cs = ["r","g","b"]

plt.figure(figsize=(10,6))
plt.plot(x1,x2,"b+",label="datapoints")

for alpha , c in zip(alphas,cs):
    preds = get_preds_ridge(x1,x2,alpha)
    plt.plot(sorted(x1[:,0]),preds[np.argsort(x1[:,0])],c,label="Alpha:{}".format(alpha))
    
plt.legend()
plt.show()

