#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pd 


# In[23]:


import pandas as pd  # Ensure this is Pandas
import matplotlib.pyplot as plt  # Correct import for Matplotlib

df = pd.read_csv("archive (7).zip")  # Now this should work if the file is a CSV or ZIP with a single CSV


# In[24]:


df.head()


# In[25]:


X = df.iloc[:,0:2].values
y = df.iloc[:,-1].values


# In[26]:


plt.scatter(X[:,0],X[:,1],c=y)


# In[27]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()


# In[28]:


clf.fit(X,y)


# In[29]:


get_ipython().system('pip install mlxtend')


# In[30]:


pip install mlxtend


# In[31]:


import mlxtend
print(mlxtend.__version__)  # This should print the installed version


# In[32]:


from mlxtend.plotting import plot_decision_regions


# In[33]:


plot_decision_regions(X,y.astype("int"),clf,legend=2)


# In[34]:


from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(clf,X,y,scoring="accuracy",cv=10))


# In[36]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3,include_bias=False)
X_trf = poly.fit_transform(X)


# In[39]:


clf1 = LogisticRegression()
np.mean(cross_val_score(clf1,X_trf,y,scoring="accuracy",cv=10))


# In[41]:


def plot_decision_boundary(X,y,degree=1):
    poly = PolynomialFeatures(degree=degree)
    X_trf = poly.fit_transform(X)
    
    clf = LogisticRegression()
    clf.fit(X_trf,y)
    
    accuracy = np.mean(cross_val_score(clf,X_trf,y,scoring="accuracy",cv=10))
    
    a=np.arange(start=X[:,0].min()-1,stop=X[:,0].max()+1,step=0.01)
    b=np.arange(start=X[:,1].min()-1,stop=X[:,1].max()+1,step=0.01)
    
    XX,YY = np.meshgrid(a,b)
    
    input_array=np.array([XX.ravel(),YY.ravel()]).T
    
    labels=clf.predict(poly.transform(input_array))
    
    plt.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.5)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.title("Degree={},accuracy is {}".format(degree,np.round(accuracy,4)))


# In[42]:


plot_decision_boundary(X,y)


# In[44]:


plot_decision_boundary(X,y,degree=2)


# In[45]:


plot_decision_boundary(X,y,degree=3)


# In[46]:


plot_decision_boundary(X,y,degree=4)


# In[47]:


plot_decision_boundary(X,y,degree=5)

