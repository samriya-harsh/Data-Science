#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

melbourne_file_path = ('melb_data.csv')
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns


# In[12]:


y = melbourne_data.Price


# In[13]:


melbourne_features= ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']


# In[17]:


X = melbourne_data[melbourne_features]


# In[18]:


X.describe() 


# In[19]:


X.info()


# In[20]:


X.isnull()


# In[22]:


X.info(5)


# In[23]:


X.head()


# In[24]:


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)


# In[26]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

