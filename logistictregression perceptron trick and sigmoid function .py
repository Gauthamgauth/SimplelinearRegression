#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import make_classification 
import numpy as np
X,y = make_classification(n_samples=100,n_features=2,n_informative=1,n_redundant=0,
                         n_classes=2,n_clusters_per_class=1,random_state=41,hypercube=False,class_sep=10)


# In[3]:


import matplotlib.pyplot as plt


# In[5]:


plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap="winter",s=100)


# In[27]:


def perceptron(X,y):
    
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    
    for i in range(1000):
        j = np.random.randint(0,100)
        y_hat = step(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]
        
    return weights[0],weights[1:]


# In[22]:


X


# In[23]:


y


# In[24]:


np.ones(3)


# In[25]:


def step(z):
    return 1 if z > 0 else 0 


# In[28]:


intercept_,coef_ = perceptron(X,y)


# In[29]:


print(coef_)
print(intercept_)


# In[32]:


m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])


# In[35]:


x_input = np.linspace(-3,3,100)
y_input = m*x_input + b 


# In[38]:


plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap="winter",s=100)
plt.ylim(-3,2)


# In[40]:


# with logistic regression, sklearn
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(X,y)


# In[41]:


m = -(lor.coef_[0][0]/lor.coef_[0][1])
b = -(lor.intercept_/lor.coef_[0][1])


# In[42]:


x_input = np.linspace(-3,3,100)
y_input = m*x_input + b


# In[43]:


plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.plot(x_input,y_input,color='black',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap="winter",s=100)
plt.ylim(-3,2)


# In[57]:


x_input1 = np.linspace(-3,3,100)
y_input1 =m * x_input + b


# In[51]:


def perceptron(X,y):
    
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    
    for i in range(1000):
        j = np.random.randint(0,100)
        y_hat = sigmoid(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]
        
    return weights[0],weights[1:]


# In[52]:


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# In[53]:


intercept_,coef_ = perceptron(X,y)


# In[54]:


m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])


# In[59]:


x_input2 = np.linspace(-3,3,100)
y_input2 = m*x_input + b


# In[60]:


plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.plot(x_input1,y_input1,color='black',linewidth=3)
plt.plot(x_input2,y_input2,color='black',linewidth=3)

plt.scatter(X[:,0],X[:,1],c=y,cmap="winter",s=100)
plt.ylim(-3,2)

