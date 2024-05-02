#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("csv_data.csv")


# In[3]:


df.info()


# In[4]:


df.head(8)


# In[5]:


df.tail(8)


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[9]:


df.columns


# In[10]:


df.drop('S.No.',inplace=True,axis=1)


# In[11]:


df["Alcohol"].unique()


# In[12]:


df["Alcohol"].fillna("Not Mentioned",inplace=True)


# In[13]:


df["Alcohol"].unique()


# In[14]:


df["State"].unique()


# In[15]:


df["Category"].unique()


# In[16]:


df["Category"] = df["Category"].str.split(" ").str[0]


# In[17]:


df["Category"].head()


# In[18]:


df = df.astype({'Category':'int'})


# In[19]:


df.info()


# In[20]:


df.rename(columns={'Category':'StarRating'},inplace=True)


# In[21]:


df.info()


# In[23]:


df["City"]


# In[24]:


df['City'] = df['City'].str.split('*').str[0]
df['City'].unique()


# In[25]:


s_names = df.State.value_counts().index
s_hotels = df.State.value_counts().values


# In[26]:


hotels = df.groupby('State').size().reset_index().rename(columns={0:'Hotels','State':'States'})
hotels


# In[27]:


matplotlib.rcParams['figure.figsize'] = (30,15)
p = sns.barplot(x='States',y='Hotels',data = hotels)
p.bar_label(p.containers[0])
plt.title("No. of Hotels in different States")
p.set_xticklabels(labels=hotels["States"], rotation=35)
sns.set(font_scale=1.25)
plt.show()


# In[29]:


ratings = df.StarRating.value_counts().index
ratings_count = df.StarRating.value_counts().values


# In[30]:


ratings


# In[31]:


matplotlib.rcParams['figure.figsize'] = (12,6)
plt.pie(ratings_count,labels=ratings,autopct="%1.2f%%")
plt.title("Star category hotel share")
plt.show()


# In[32]:


rating = df.groupby('StarRating').size().reset_index()
rating


# In[33]:


h_in_city = df.groupby("City").size().reset_index().rename(columns={0:'Hotels'})
h_in_city = h_in_city.sort_values(by='Hotels',ascending=False) #sorting based on no. of hotels


# In[34]:


h_in_city[0:9]


# In[35]:


matplotlib.rcParams['figure.figsize'] = (30,15)
p = sns.barplot(x='City',y='Hotels',data = h_in_city[0:9])
p.bar_label(p.containers[0])
plt.title("No. of Hotels top 10 cities")
p.set_xticklabels(labels=h_in_city["City"][0:9], rotation=35)
sns.set(font_scale=1.25)
plt.show()


# In[36]:


df.info()


# In[37]:


s_rooms = df.groupby('State')['Total Rooms'].sum().reset_index()
s_rooms.head()


# In[38]:


matplotlib.rcParams['figure.figsize'] = (30,15)
p = sns.barplot(x='State',y='Total Rooms',data = s_rooms)
p.bar_label(p.containers[0])
plt.title("No. of Rooms in each state")
p.set_xticklabels(labels=s_rooms["State"], rotation=35)
sns.set(font_scale=1.25)
plt.show()

