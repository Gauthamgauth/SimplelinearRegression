#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import make_classification 
import numpy as np

X,y = make_classification(n_samples=100,n_features=2,n_informative=1,n_redundant=0,
                         n_classes=2,n_clusters_per_class=1,random_state=41,hypercube=False,class_sep=20)


# In[5]:


import matplotlib.pyplot as plt


# In[9]:


plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap="winter",s=100)


# In[12]:


from sklearn.linear_model import LogisticRegression
lor = LogisticRegression(penalty = None,solver="sag")
lor.fit(X,y)


# In[13]:


print(lor.coef_)
print(lor.intercept_)


# In[16]:


m1 = -(lor.coef_[0][0]/lor.coef_[0][1])
b1 = -(lor.intercept_/lor.coef_[0][1])


# In[17]:


x_input = np.linspace(-3,3,100)
y_input = m1*x_input + b1


# In[28]:


import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gd(X, y):
    X = np.insert(X, 0, 1, axis=1)  # Adding bias term
    weights = np.ones(X.shape[1])   # Initialize weights correctly
    lr = 0.5  # Learning rate

    for i in range(2500):
        y_hat = sigmoid(np.dot(X, weights))  # Fixed variable name
        weights = weights + lr * (np.dot(X.T, (y - y_hat)) / X.shape[0])  # Fixed update rule

    return weights[1:], weights[0]  # Returning weights and bias separately


# In[29]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[31]:


coef_,intercept_ = gd(X,y)


# In[32]:


m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])


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




