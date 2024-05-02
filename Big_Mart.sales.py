#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
test1 = test.copy()
train.shape,test.shape


# In[3]:


train.head()


# In[4]:


test["Outlet_Size"].unique()


# In[5]:


train.nunique()


# In[6]:


test.nunique()


# In[7]:


train.isna().sum()


# In[8]:


map1 = {"Small":1,"Medium":2,"High":3}
train["Outlet_Size"] = train["Outlet_Size"].map(map1)
train["Item_Weight"] = train["Item_Weight"].fillna(train.Item_Weight.mean())
train["Outlet_Size"] = train["Outlet_Size"].fillna(train["Outlet_Size"].median())


# In[9]:


train.isna().sum()


# In[10]:


map1 = {"Small":1,"Medium":2,"High":3}
test["Outlet_Size"] = test["Outlet_Size"].map(map1)
test["Item_Weight"] = test["Item_Weight"].fillna(test.Item_Weight.mean())
test["Outlet_Size"] = test["Outlet_Size"].fillna(test["Outlet_Size"].median())


# In[11]:


train.head()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10,10)
plt.hist(train["Item_Outlet_Sales"],bins = 100)
# plt.show()

# plt.rcParams['figure.figsize'] = (10,10)
plt.hist(train["Item_MRP"],alpha = 0.3,bins = 150)
plt.show()


# In[13]:


plt.rcParams['figure.figsize'] = (5,5)
plt.hist(train["Item_MRP"],alpha = 0.3,bins = 150)
plt.show()


# In[15]:


sns.countplot(train["Outlet_Size"],palette = 'dark')
plt.show()


# In[17]:


sns.violinplot(x=train["Outlet_Size"],y=train["Item_Outlet_Sales"],hue = train["Outlet_Size"],palette = "Reds")
plt.legend()
plt.show()


# In[18]:


train.drop(labels = ["Outlet_Establishment_Year"],inplace = True,axis =1)
test.drop(labels = ["Outlet_Establishment_Year"],inplace = True,axis =1)


# In[19]:


feat = ['Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content',"Item_Type"]
X = pd.get_dummies(train[feat])
train = pd.concat([train,X],axis=1)


# In[20]:


train.head()


# In[21]:


feat = ['Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content',"Item_Type"]
X1 = pd.get_dummies(test[feat])
test = pd.concat([test,X1],axis=1)


# In[22]:


train.drop(labels = ["Outlet_Size",'Outlet_Location_Type',"Outlet_Type",'Item_Fat_Content','Outlet_Identifier','Item_Identifier',"Item_Type"],axis=1,inplace = True)
test.drop(labels = ["Outlet_Size",'Outlet_Location_Type',"Outlet_Type",'Item_Fat_Content','Outlet_Identifier','Item_Identifier',"Item_Type"],axis=1,inplace = True)


# In[23]:


X_train = train.drop(labels = ["Item_Outlet_Sales"],axis=1)
y_train = train["Item_Outlet_Sales"]
X_train.shape,y_train.shape


# In[24]:


train.head()


# In[25]:


y_train.head()


# In[26]:


from sklearn import preprocessing

x = X_train.values #returns a numpy array
test_s = test.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled_train = min_max_scaler.fit_transform(x)
x_scaled_test = min_max_scaler.fit_transform(test_s)
df_train = pd.DataFrame(x_scaled_train)
df_test = pd.DataFrame(x_scaled_test)


# In[27]:


df_train.head()


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.4)


# In[29]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[30]:


preds = model.predict(X_test)


# In[31]:


plt.scatter(y_test, preds)
plt.show()


# In[32]:


sns.distplot((y_test-preds),bins=50)
plt.show()


# In[33]:


from sklearn import metrics
print("MAE:", metrics.mean_absolute_error(y_test, preds))
print('MSE:', metrics.mean_squared_error(y_test, preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))


# In[34]:


predictions = model.predict(df_test)
final = pd.DataFrame({"Item_Identifier":test1["Item_Identifier"],"Outlet_Identifier":test1["Outlet_Identifier"],"Item_Outlet_Sales":abs(predictions)})
final.head()


# In[36]:


final.to_csv('Submiss1.csv',index=False,header=True)

