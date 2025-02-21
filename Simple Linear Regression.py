#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[9]:


df = pd.read_csv("archive (3).zip")


# In[10]:


df.head()


# In[13]:


plt.scatter(df["cgpa"],df["package"])
plt.xlabel("CGPA")
plt.ylabel("LPA")


# In[18]:


X = df.iloc[:,0:1]
y = df.iloc[:,-1]


# In[19]:


X


# In[20]:


y


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


lr = LinearRegression()


# In[26]:


lr.fit(X_train,y_train)


# In[27]:


X_test


# In[28]:


y_test


# In[34]:


lr.predict(X_test.iloc[2].values.reshape(1,1))


# In[35]:


lr.predict(X_test.iloc[5].values.reshape(1,1))


# In[39]:


plt.scatter(df["cgpa"],df["package"])
plt.plot(X_train,lr.predict(X_train),color="red")
plt.xlabel("CGPA")
plt.ylabel("LPA")


# In[41]:


m = lr.coef_


# In[43]:


b =lr.intercept_


# In[44]:


m * 8.58 + b 


# In[45]:


m * 5.88 + b

