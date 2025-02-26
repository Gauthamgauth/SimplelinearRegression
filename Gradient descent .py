#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_regression


# In[2]:


import numpy as np


# In[5]:


X,y = make_regression(n_samples=4,n_features=1,n_informative=1,n_targets=1,noise=80,random_state=13)


# In[7]:


import matplotlib.pyplot as plt
plt.scatter(X,y)


# In[8]:


# ordinary least square
from sklearn.linear_model import LinearRegression


# In[9]:


reg = LinearRegression()
reg.fit(X,y)


# In[10]:


reg.coef_


# In[11]:


reg.intercept_


# In[12]:


plt.scatter(X,y)
plt.plot(X,reg.predict(X),color="red")


# In[13]:


# applying gradient descent assuming slope is constant m = 78.35
# and assue the starting value of intercept b = 0
y_pred = ((78.35* X)+0).reshape(4)


# In[40]:


plt.scatter(X,y)
plt.plot(X,reg.predict(X),color="red",label="OLS")
plt.plot(y_pred,color="#00a65a",label="b=0")
plt.legend()
plt.show()


# In[41]:


m=78.35
b=0

loss_slope = -2 * np.sum(y - m*X.ravel()-b)
loss_slope


# In[42]:


# learning rate ar 0.1 

lr = 0.1
step_size = loss_slope*lr
step_size


# In[43]:


# calculating the nre intercept 
b = b - step_size
b


# In[44]:


y_pred1 = ((78.35 * X)+b).reshape(4)

plt.scatter(X,y)
plt.plot(X,reg.predict(X),color="red",label="OLS")
plt.plot(X,y_pred1,color="#00a65a",label="b={}".format(b))
plt.plot(X,y_pred,color="#A3E4D7",label="b=0")
plt.legend()
plt.show()


# In[46]:


# again doing the slope

loss_slope = -2 *np.sum(y-m*X.ravel()-b)
loss_slope


# In[47]:


step_size = loss_slope*lr
step_size


# In[48]:


b = b-step_size
b


# In[49]:


y_pred2 = ((78.35 * X)+b).reshape(4)

plt.scatter(X,y)
plt.plot(X,reg.predict(X),color="red",label="OLS")
plt.plot(X,y_pred2,color="#00a65a",label="b={}".format(b))
plt.plot(X,y_pred1,color="#A3E4D7",label="b={}".format(b))
plt.plot(X,y_pred,color="#A3E4D7",label="b=0")
plt.legend()
plt.show()


# In[50]:


loss_slope = -2 *np.sum(y-m*X.ravel()-b)
loss_slope


# In[51]:


step_size = loss_slope*lr
step_size


# In[52]:


b = b-step_size
b


# In[53]:


y_pred3 = ((78.35 * X)+b).reshape(4)

plt.scatter(X,y)
plt.plot(X,reg.predict(X),color="red",label="OLS")
plt.plot(X,y_pred3,color="#00a65a",label="b={}".format(b))
plt.plot(X,y_pred2,color="#00a65a",label="b={}".format(b))
plt.plot(X,y_pred1,color="#00a65a",label="b={}".format(b))
plt.plot(X,y_pred,color="#A3E4D7",label="b={}".format(b))
plt.plot(X,y_pred,color="#A3E4D7",label="b=0")
plt.legend()
plt.show()


# In[54]:


loss_slope = -2 *np.sum(y-m*X.ravel()-b)
loss_slope


# In[55]:


step_size = loss_slope*lr
step_size


# In[56]:


b = b-step_size
b


# In[57]:


y_pred4 = ((78.35 * X)+b).reshape(4)

plt.scatter(X,y)
plt.plot(X,reg.predict(X),color="red",label="OLS")
plt.plot(X,y_pred4,color="#00a65a",label="b={}".format(b))
plt.plot(X,y_pred3,color="#00a65a",label="b={}".format(b))
plt.plot(X,y_pred2,color="#00a65a",label="b={}".format(b))
plt.plot(X,y_pred1,color="#00a65a",label="b={}".format(b))
plt.plot(X,y_pred,color="#A3E4D7",label="b={}".format(b))
plt.plot(X,y_pred,color="#A3E4D7",label="b=0")
plt.legend()
plt.show()


# In[58]:


loss_slope = -2 *np.sum(y-m*X.ravel()-b)
loss_slope


# In[59]:


step_size = loss_slope*lr
step_size


# In[60]:


b = b-step_size
b

