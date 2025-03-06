#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


# In[3]:


X = 6 * np.random.rand(200,1)-3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.rand(200,1)

# y = 0.8x^2 + 0.9x + 2 polynoimial equation


# In[4]:


plt.plot(X,y,"b.")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


# In[5]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[6]:


lr = LinearRegression()


# In[7]:


lr.fit(X_train,y_train)


# In[13]:


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(r2)  # This should give you 0.203865909


# In[15]:


plt.plot(X_train,lr.predict(X_train),color="r")
plt.plot(X,y,"b.")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


# In[21]:


# applying polynomial regression, degree=2 
poly = PolynomialFeatures(degree=2,include_bias=False)

X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)


# In[22]:


print(X_train[0])
print(X_train_trans[0])


# In[23]:


poly = PolynomialFeatures(degree=2,include_bias=True)

X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)


# In[24]:


print(X_train[0])
print(X_train_trans[0])


# In[25]:


lr= LinearRegression()
lr.fit(X_train_trans,y_train)


# In[26]:


y_pred = lr.predict(X_test_trans)


# In[27]:


r2_score=(y_test,y_pred)


# In[29]:


print(r2)


# In[30]:


print(lr.coef_)
print(lr.intercept_)


# In[32]:


X_new=np.linspace(-3,3,200).reshape(200,1)
X_new_poly = poly.transform(X_new)
y_new = lr.predict(X_new_poly)


# In[33]:


plt.plot(X_new,y_new,"r-",linewidth=2,label="predictions")
plt.plot(X_train,y_train,"b.",label="training points")
plt.plot(X_test,y_test,"g.",label="testing points ")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


# In[43]:


def polynomial_regression(degree, X, y, X_train, y_train, X_test, y_test):
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)

    # Define polynomial features transformation
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Standardize the features
    std_scaler = StandardScaler()
    
    # Linear Regression model
    lin_reg = LinearRegression()
    
    # Create a pipeline for polynomial regression
    polynomial_regression = Pipeline([
        ("poly_features", polybig_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg),
    ])
    
    # Fit the model to the training data
    polynomial_regression.fit(X, y)
    
    # Predict new values
    y_newbig = polynomial_regression.predict(X_new)
    
    # Plot the polynomial regression line
    plt.plot(X_new, y_newbig, "r", label="Degree " + str(degree), linewidth=2)
    
    # Plot the training and test points
    plt.plot(X_train, y_train, "b.", markersize=8, label="Training data")
    plt.plot(X_test, y_test, "g.", markersize=8, label="Test data")
    
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-3, 3, 0, 10])
    plt.show()

    


# In[45]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)  # For reproducibility
X = np.random.rand(100, 1) * 6 - 3  # Random values between -3 and 3
y = 2 + X + X**2 + np.random.randn(100, 1) * 0.5  # Quadratic relationship with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def polynomial_regression(degree, X, y, X_train, y_train, X_test, y_test):
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)  # New X values for plotting

    # Create polynomial features
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Standardize the features
    std_scaler = StandardScaler()
    
    # Linear Regression model
    lin_reg = LinearRegression()
    
    # Create a pipeline for polynomial regression
    polynomial_regression = Pipeline([
        ("poly_features", polybig_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg),
    ])
    
    # Fit the model to the training data
    polynomial_regression.fit(X_train, y_train)
    
    # Predict new values
    y_newbig = polynomial_regression.predict(X_new)
    
    # Plot the polynomial regression line
    plt.plot(X_new, y_newbig, "r", label="Degree " + str(degree), linewidth=2)
    
    # Plot the training and test points
    plt.plot(X_train, y_train, "b.", markersize=8, label="Training data")
    plt.plot(X_test, y_test, "g.", markersize=8, label="Test data")
    
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-3, 3, 0, 10])
    plt.show()

# Now, call the function with all required arguments
polynomial_regression(2, X, y, X_train, y_train, X_test, y_test)


# In[47]:


x = 7 * np.random.rand(100,1)- 2.8
y = 7 * np.random.rand(100,1) - 2.8

z = x**2 + y**2 + 0.2*x , 0.2*y + 0.1*x*y + 2 + np.random.rand(100,1)


# In[50]:


import numpy as np
import plotly.express as px

df = px.data.iris()

# Convert x, y, z to NumPy arrays before calling .ravel()
x, y, z = np.array(df["sepal_length"]), np.array(df["sepal_width"]), np.array(df["petal_length"])

fig = px.scatter_3d(df, x=x.ravel(), y=y.ravel(), z=z.ravel())
fig.show()

