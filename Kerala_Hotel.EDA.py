#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


df = pd.read_csv('munnar.csv')
df.head()


# In[9]:


df.info()


# In[10]:


df.tail()


# In[11]:


df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


df.rename(columns={'Hotel Name\t\t\t\t\t\t\t\t\t':'Hotel Name','Rating Description':'Rating_Description'},inplace=True)


# In[15]:


df["Star Rating"]


# In[16]:


#filling nan with mean value
df["Star Rating"].fillna(value=df["Star Rating"].mean(),inplace=True)


# In[17]:


#converting to int
df = df.astype({'Star Rating':'int'})
df["Star Rating"].unique()


# In[18]:


df["Tax"]


# In[19]:


df["Tax"] = df["Tax"].str.replace(",","") #removes all commas at once in a column


# In[20]:


df["Tax"].unique()


# In[21]:


#filling nan with mean value
df["Tax"].fillna(value="0",inplace=True)
df = df.astype({'Tax':'int'})
df.replace([0],df["Tax"].mean(),inplace=True)


# In[22]:


df = df.astype({'Tax':'int'})


# In[23]:


df["Price"].unique()


# In[25]:


df["Price"] = df["Price"].str.replace(",","") #removes all commas at once in a column
df = df.astype({'Price':'int'})


# In[26]:


df.info()


# In[27]:


df["Distance to Landmark"].unique()


# In[28]:


sns.heatmap(df[['Rating','Reviews','Star Rating','Price','Tax']].corr(),annot=True,cmap='coolwarm')
plt.title("Heatmap")
plt.show()


# In[29]:


plt.title("Rating Distribution")
df["Rating"].plot(kind='hist',edgecolor='black')
plt.show()


# In[30]:


df["Price"].mean()


# In[31]:


plt.title("Rating Vs Reviews")
plt.scatter(df["Rating"],df["Reviews"],color='purple')
plt.xlabel("Rating")
plt.ylabel("Reviews")
plt.grid(True)
plt.show()


# In[32]:


plt.scatter(df["Star Rating"],df["Price"])
plt.title("Star Rating")
plt.grid(True)
plt.show()


# In[33]:


df_s_hotels = df.groupby('Star Rating').size().reset_index().rename(columns={0:"Count"})
sns.barplot(x='Star Rating',y='Count',data=df_s_hotels)


# In[34]:


df_loc_price = df.groupby('Location')['Price'].mean().reset_index()
df_loc_price = df_loc_price.astype({'Price':'int'})
df_loc_price.head()


# In[35]:


matplotlib.rcParams['figure.figsize'] = (40,20)
plt.title("Average price in each location")
p=sns.barplot(x='Location',y='Price',data=df_loc_price)
p.bar_label(p.containers[0])
p.set_xticklabels(labels=df_loc_price['Location'],rotation=70)
plt.show()
sns.set(font_scale=2)


# In[36]:


sns.set(font_scale=1.5)
matplotlib.rcParams['figure.figsize'] = (12,6)
r_desc = df.Rating_Description.value_counts().index
r_count = df.Rating_Description.value_counts().values
plt.pie(r_count,labels=r_desc,autopct="%1.2f%%")
plt.title("Hotels based on Rating Description")
plt.show()

