#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[7]:


housing = pd.read_csv('data.csv')


# In[8]:


type(housing)


# In[10]:


housing.head(10)


# In[11]:


housing.tail(5)


# In[12]:


housing.info()


# In[13]:


housing['CHAS'].value_counts()


# In[14]:


housing.describe()


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


# # For plotting histogram
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))


# In[17]:


# For learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[18]:


train_set, test_set = split_train_test(housing, 0.2)


# In[19]:


print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[20]:


from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[21]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[22]:


strat_test_set['CHAS'].value_counts()


# In[23]:


strat_train_set['CHAS'].value_counts()


# In[24]:


housing = strat_train_set.copy()


# In[25]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[26]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[27]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# In[28]:


housing["TAXRM"] = housing['TAX']/housing['RM']


# In[29]:


housing.head()


# In[30]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[31]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[32]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# In[33]:


# To take care of missing attributes, you have three options:
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)


# In[34]:


a = housing.dropna(subset=["RM"]) #Option 1
a.shape
# Note that the original housing dataframe will remain unchanged


# In[35]:


housing.drop("RM", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged


# In[37]:


median = housing["RM"].median() # Compute median for Option 3


# In[38]:


housing["RM"].fillna(median) # Option 3
# Note that the original housing dataframe will remain unchanged


# In[39]:


housing.shape


# In[40]:


housing.describe() # before we started filling missing attributes


# In[41]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[42]:


imputer.statistics_


# In[43]:


X = imputer.transform(housing)


# In[44]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[45]:


housing_tr.describe()


# In[46]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[47]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[48]:


housing_num_tr.shape


# In[49]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[50]:


some_data = housing.iloc[:5]


# In[51]:


some_labels = housing_labels.iloc[:5]


# In[52]:


prepared_data = my_pipeline.transform(some_data)


# In[53]:


model.predict(prepared_data)


# In[54]:


list(some_labels)


# In[55]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[56]:


rmse


# In[57]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[58]:


rmse_scores


# In[59]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[60]:


print_scores(rmse_scores)


# In[61]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# In[62]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[63]:


final_rmse


# In[1]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)

