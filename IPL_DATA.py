#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("matches.csv")
df.head(5)


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[8]:


df.tail(10)


# In[14]:


def name_changer(data):
    if data=='Delhi Daredevils':
        data='Delhi Capitals'
    elif data=='Deccan Chargers':
        data='Sunrisers Hyderabad'
    elif data=='Pune Warriors' or data=='Rising Pune Supergiant':
        data='Rising Pune Supergiants'
    return data    
    


# In[18]:


df['team1']=df['team1'].apply(lambda x:name_changer(x))
df['team2']=df['team2'].apply(lambda x:name_changer(x))
df['toss_winner']=df['toss_winner'].apply(lambda x:name_changer(x))
df['winner']=df['winner'].apply(lambda x:name_changer(x))


# In[19]:


df.winner.value_counts()


# In[21]:


plt.figure(figsize=(15,5))
match_wins=df.winner.value_counts()
df.winner.value_counts().plot.bar()


# In[22]:


tot_match=df.team1.value_counts()+df.team2.value_counts()
tot_match.sort_values(ascending=False)


# In[23]:


plt.figure(figsize=(15,5))
tot_match.sort_values(ascending=False).plot.bar()


# In[24]:


win_per=(match_wins/tot_match)*100
win_per.sort_values(ascending=False)


# In[25]:


plt.figure(figsize=(15,5))

win_per.sort_values(ascending=False).plot.bar()


# In[27]:


df.player_of_match.value_counts().head(20)


# In[28]:


df.venue.value_counts()


# In[30]:


plt.figure(figsize=(15,5))
df.venue.value_counts().plot.bar()


# In[33]:


um=df.umpire1.value_counts()+df.umpire2.value_counts()
um.sort_values(ascending=False).head(10)


# In[34]:


plt.figure(figsize=(15,5))
um.sort_values(ascending=False).head(15).plot.bar()

