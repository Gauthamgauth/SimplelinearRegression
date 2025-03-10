#!/usr/bin/env python
# coding: utf-8

# In[5]:


# how coefficient affected 

from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# In[6]:


data = load_diabetes()


# In[9]:


df = pd.DataFrame(data.data,columns = data.feature_names)
df["TARGETS"] = data.target


# In[10]:


df.head()


# In[11]:


df.shape


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

data = load_diabetes()
X = data.data  # Feature matrix
y = data.target  # Target values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[17]:


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


# In[22]:


coefs = []  # Initialize the list before appending values

for i in range(1, 10):
    reg = Ridge(alpha=i)
    reg.fit(X_train, y_train)
    
    coefs.append(reg.coef_.tolist())  # Now 'coefs' exists

    y_pred = reg.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))  # 'r2_scores' should also be initialized



# In[23]:


r2_scores = []


# In[28]:


plt.figure(figsize=(14,9))
plt.subplot(221)
plt.bar(data.feature_names,coefs[0])
plt.title("Alpha = 0","r2_score = {}".format(round(r2_score[0],2)))

plt.subplot(222)
plt.bar(data.feature_names,coefs[0])
plt.title("Alpha = 0","r2_score = {}".format(round(r2_score[1],2)))

plt.subplot(223)
plt.bar(data.feature_names,coefs[0])
plt.title("Alpha = 0","r2_score = {}".format(round(r2_score[2],2)))

plt.subplot(224)
plt.bar(data.feature_names,coefs[0])
plt.title("Alpha = 0","r2_score = {}".format(round(r2_score[3],2)))


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




