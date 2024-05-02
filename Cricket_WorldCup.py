#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[42]:


Batsman_Data = pd.read_csv('Batsman_Data.csv.zip')
Ground_Data = pd.read_csv('Ground_Averages.csv')
ODI_Scores_Data = pd.read_csv('ODI_Match_Totals.csv')
ODI_Results_Data = pd.read_csv('ODI_Match_Results.csv')
WC_Players_Data = pd.read_csv('WC_players.csv')
Bowler_Data = pd.read_csv('Bowler_data.csv.zip')


# In[43]:


Ground_Data.sample(10)


# In[44]:


Batsman_Data.isnull().sum()


# In[45]:


Batsman_Data.info()


# In[46]:


Batsman_Data.head(10)


# In[47]:


Ground_Data.isnull().sum()


# In[48]:


Ground_Data.info()


# In[49]:


Ground_Data.head(8)


# In[50]:


Ground_Data.describe()


# In[51]:


Ground_Data.isnull().sum()


# In[52]:


ODI_Results_Data.head()


# In[53]:


ODI_Results_Data.isnull().sum()


# In[54]:


ODI_Results_Data.info()


# In[55]:


ODI_Scores_Data.head()


# In[56]:


ODI_Results_Data.info()


# In[57]:


WC_Players_Data.info()


# In[58]:


WC_Players_Data.isnull().sum


# In[59]:


WC_Players_Data.tail(30)


# In[60]:


ODI_Scores_Data["Scores_ID"] = ODI_Scores_Data["Unnamed: 0"]
ODI_Scores_Data.drop(columns="Unnamed: 0",inplace=True)


# In[61]:


WC_venue_pitches = ["The Oval, London","Trent Bridge, Nottingham","Sophia Gardens, Cardiff","County Ground, Bristol","Rose Bowl, Southampton","County Ground, Taunton","Old Trafford, Manchester","Edgbaston, Birmingham","Headingley, Leeds","Lord's, London","Riverside Ground, Chester-le-Street"]


# In[62]:


#Total Grounds
WC_Ground_Stats = []
ODI_Grounds = ODI_Scores_Data.Ground
for i in ODI_Grounds:
    for j in WC_venue_pitches:
        if i in j:
            #print("i ; ",i,"--j : ",j)
            WC_Ground_Stats.append((i,j))
            


# In[63]:


Ground_names = dict(set(WC_Ground_Stats))
def Full_Ground_names(value):
    return Ground_names[value]
Ground_names


# In[64]:


#Let's gather the data of all ODI's in these WC Venues
WC_Grounds_History = ODI_Scores_Data[ODI_Scores_Data.Ground.isin([Ground[0] for Ground in WC_Ground_Stats])]
WC_Grounds_History["Ground"] = WC_Grounds_History.Ground.apply(Full_Ground_names)
WC_Grounds_History.head()


# In[65]:


WC_Grounds_History.sample(5)


# In[66]:


WC_Grounds_History.Result.value_counts()


# In[67]:


WC_Grounds_History = WC_Grounds_History[~WC_Grounds_History.Result.isin(["-"])]
WC_Grounds_History.Result.value_counts()


# In[69]:


WC_Grounds_History.sample(5)

