#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[4]:


df= pd.read_csv("archive (3).zip")


# In[ ]:





# In[5]:


df.head()


# In[7]:


plt.scatter(df["cgpa"],df["package"])
plt.xlabel("CGPA")
plt.ylabel("LPA")


# In[8]:


X = df.iloc[:,0:1]
y = df.iloc[:,-1]


# In[9]:


X


# In[10]:


y


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=2)


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


lr = LinearRegression()


# In[15]:


lr.fit(X_train,y_train)


# In[18]:


plt.scatter(df["cgpa"],df["package"])
plt.plot(X_train,lr.predict(X_train),color="red")
plt.xlabel = ("CGPA")
plt.ylabel = ("LPA")


# In[20]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[27]:


y_pred =lr.predict(X_test)


# In[28]:


y_test.values


# In[30]:


print("MAE",mean_absolute_error(y_test,y_pred))


# In[31]:


print("MSE",mean_squared_error(y_test,y_pred))


# In[33]:


print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))


# In[35]:


print("MSE",r2_score(y_test,y_pred)) 


# In[38]:


print("MSE",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)


# In[39]:


#adjusted r2 score
X_test.shape


# In[40]:


1-((1-r2)*(40-1)/(40-1-1))

