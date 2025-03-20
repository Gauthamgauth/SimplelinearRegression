#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import recall_score,precision_score,f1_score


# In[12]:


import pandas as pd 


# In[14]:


cdf = pd.DataFrame(confusion_matrix(y_test,y_pred1),columns=list(range(0,2)))


# In[13]:


print(precision_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))
print(f1_score(y_test,y_pred1))


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




