#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.datasets import load_diabetes

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split


# In[19]:


data = load_diabetes()

df = pd.DataFrame(data.data,columns=data.feature_names)
df["TARGET"] = data.target

df.head()


# In[25]:


X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2,random_state=2)


# In[27]:


coefs = []
r2_scores = []

for i in[0,0.1,1,10]:
    reg = Lasso(alpha=i)
    reg.fit(X_train,y_train)
    
    coefs.append(reg.coef_.tolist())
    y_pred = reg.predict(X_test)
    r2_scores.append(r2_score(y_test,y_pred))


# In[29]:


plt.figure(figsize=(14,9))
plt.subplot(221)
plt.bar(data.feature_names,coefs[0])
plt.title("Alpha = 0,r2_score={}".format(round(r2_scores[0],2)))

plt.subplot(222)
plt.bar(data.feature_names,coefs[1])
plt.title("Alpha = 0.1,r2_score={}".format(round(r2_scores[1],2)))

plt.subplot(223)
plt.bar(data.feature_names,coefs[2])
plt.title("Alpha = 1,r2_score={}".format(round(r2_scores[2],2)))

plt.subplot(223)
plt.bar(data.feature_names,coefs[3])
plt.title("Alpha = 10,r2_score={}".format(round(r2_scores[3],2)))


# In[30]:


alphas = [0,0.0001,0.001,0.01,0.1,1,10.100,1000,10000]

coefs = []

for i in alphas:
    reg = Lasso(alpha=i)
    reg.fit(X_train,y_train)
    
    coefs.append(reg.coef_.tolist())
    
    


# In[32]:


input_array = np.array(coefs)

coef_df = pd.DataFrame(input_array,columns=data.feature_names)
coef_df["alpha"] = alphas
coef_df.set_index("alpha")


# In[34]:


alphas = [0,0.0001,0.0005,0.001,0.005,0.1,0.5,1,5,10]

coefs = []
for i in alphas:
    reg = Lasso(alpha=i)
    reg.fit(X_train,y_train)
    
    coefs.append(reg.coef_.tolist())


# In[36]:


input_array = np.array(coefs).T

plt.figure(figsize=(15,8))
plt.plot(alphas,np.zeros(len(alphas)),color="black",linewidth=5)

for i in range(input_array.shape[0]):
    plt.plot(alphas,input_array[i],label=data.feature_names[i])
    
plt.legend()


# In[44]:


# impact on bias variance 
m = 100
X = 5*np.random.rand(m,1)-2
y = 0.7 * X ** 2 -2 * X + 3 + np.random.randn(m,1)

plt.scatter(X,y)
plt.show()


# In[45]:


X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2,random_state=2)


# In[50]:


pip install mlxtend


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




