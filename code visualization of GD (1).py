#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_regression 
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# In[4]:


X,y = make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=20,random_state=13)


# In[5]:


plt.scatter(X,y)


# In[6]:


b = -120
m = 100
lr = 0.001
all_b=[]
all_m =[]
all_cost = []

epochs = 30

for i in range(epochs):
    slope_b=0
    slope_m=0
    cost=0
    
    for j in range(X.shape[0]):
        slope_b = slope_b - 2*(y[j] - (m*X[j])-b)
        slope_m = slope_b - 2*(y[j] - (m*X[j])-b)*X[j]
        cost = cost +(y[j]-m*X[j]-b **2)
        
        
    b = b - (lr * slope_b)
    m = m - (lr * slope_m)
    all_b.append(b)
    all_m.append(m)
    all_cost.append(cost)


# In[10]:


fig, ax = plt.subplots(figsize=(9,5))

x_i = np.arange(-3,3,0.1)
y_i = x_i*(-27) -150
ax.scatter(X,y)
line, = ax.plot(x_i,x_i*50-4,"r-",linewidth=2)


def update(i):
    label = "epoch{0}".format(i+1)
    line.set_ydata(x_i*all_m[i] + all_b[i])
    ax.set_xlabel(label)
    
anim = FuncAnimation(fig,update,repeat=True,frames=epochs,interval=500)

