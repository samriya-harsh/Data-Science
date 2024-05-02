#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np 
import pandas as pd 
import os

# Visualisation libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go


# In[7]:


url='city_day.csv.zip'
city_day_data=pd.read_csv(url)

# Extract delhi's data 

delhi_data=city_day_data.groupby('City').get_group('Delhi')


# In[9]:


delhi_data.head(8)


# In[10]:


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[11]:


missing_values_table(delhi_data)


# In[12]:


delhi_data.interpolate(limit_direction="both",inplace=True)


# In[13]:


missing_values_table(delhi_data)


# In[14]:


delhi_data['AQI_Bucket'].iloc[0]


# In[15]:


for i,each in enumerate(delhi_data['AQI_Bucket']):
    if pd.isnull(delhi_data['AQI_Bucket'].iloc[i]):
        if delhi_data['AQI'].iloc[i]>=0.0 and delhi_data['AQI'].iloc[i]<=50.0:
            delhi_data['AQI_Bucket'].iloc[i]='Good'
        elif delhi_data['AQI'].iloc[i]>=51.0 and delhi_data['AQI'].iloc[i]<=100.0:
            delhi_data['AQI_Bucket'].iloc[i]='Satisfactory'
        elif delhi_data['AQI'].iloc[i]>=101.0 and delhi_data['AQI'].iloc[i]<=200.0:
            delhi_data['AQI_Bucket'].iloc[i]='Moderate'
        elif delhi_data['AQI'].iloc[i]>=201.0 and delhi_data['AQI'].iloc[i]<=300.0:
            delhi_data['AQI_Bucket'][i]='Poor'
        elif delhi_data['AQI'].iloc[i]>=301.0 and delhi_data['AQI'].iloc[i]<=400.0:
            delhi_data['AQI_Bucket'].iloc[i]='Very Poor'
        else:
            delhi_data['AQI_Bucket'].iloc[i]='Severe'


# In[16]:


delhi_data.head(2)


# In[17]:


fig = px.line(delhi_data, x="Date", y="PM2.5")
fig.show()


# In[18]:


df_year_19_20=delhi_data[delhi_data['Date']>='2019']


# In[19]:


fig = px.line(df_year_19_20, x="Date", y="PM2.5")
fig.show()


# In[20]:


Mar_may_2019=delhi_data[(delhi_data['Date'] >= '2019-03') & (delhi_data['Date'] <= '2019-05')]
Mar_may_2020=delhi_data[(delhi_data['Date'] >= '2020-03') & (delhi_data['Date'] <= '2020-05')]


# In[21]:


plt.style.use('fivethirtyeight')


# In[22]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=Mar_may_2019['Date'], y=Mar_may_2019['PM2.5'],
                    mode='lines+markers',
                    name='PM2.5 levels of 2019'))
fig.add_trace(go.Scatter(x=Mar_may_2019['Date'], y=Mar_may_2020['PM2.5'],
                    mode='lines+markers',
                    name='PM2.5 levels of 2020'))

fig.show()


# In[23]:


fig = px.line(delhi_data, x="Date", y="PM10")
fig.show()


# In[24]:


fig = px.line(df_year_19_20, x="Date", y="PM10")
fig.show()


# In[25]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=Mar_may_2019['Date'], y=Mar_may_2019['PM10'],
                    mode='lines+markers',
                    name='PM10 levels of delhi march and april month of 2019'))
fig.add_trace(go.Scatter(x=Mar_may_2019['Date'], y=Mar_may_2020['PM10'],
                    mode='lines+markers',
                    name='PM10 levels of delhi march and april month of 2020'))
fig.show()


# In[26]:


fig = px.line(delhi_data, x="Date", y="NO")
fig.show()


# In[27]:


fig = px.line(df_year_19_20, x="Date", y="NO")
fig.show()


# In[28]:


march24_2019=delhi_data[(delhi_data['Date'] >= '2019-03-23') & (delhi_data['Date'] <= '2019-04-15')]
march24_2020=delhi_data[(delhi_data['Date'] >= '2020-03-23') & (delhi_data['Date'] <= '2020-04-15')]


# In[29]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2019['NO'],
                    mode='lines+markers',
                    name='NO levels of delhi from 23-march-2019  to 15-april 2019'))
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2020['NO'],
                    mode='lines+markers',
                    name='NO levels of delhi from 23-march-2020  to 15-april 2020'))
fig.show()


# In[30]:


fig = px.line(delhi_data, x="Date", y="NO2")
fig.show()


# In[31]:


fig = px.line(df_year_19_20, x="Date", y="NO2")
fig.show()


# In[32]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2019['NO2'],
                    mode='lines+markers',
                    name='NO2 levels of delhi from 23-march-2019  to 15-april 2019'))
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2020['NO2'],
                    mode='lines+markers',
                    name='NO2 levels of delhi from 23-march-2020  to 15-april 2020'))
fig.show()


# In[33]:


fig = px.line(delhi_data, x="Date", y="O3")
fig.show()


# In[34]:


fig = px.line(df_year_19_20, x="Date", y="O3")
fig.show()


# In[35]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2019['O3'],
                    mode='lines+markers',
                    name='O3 levels of delhi from 23-march-2019  to 15-april 2019'))
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2020['O3'],
                    mode='lines+markers',
                    name='O3 levels of delhi from 23-march-2020  to 15-april 2020'))
fig.show()


# In[36]:


fig = px.line(delhi_data, x="Date", y="SO2")
fig.show()


# In[37]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2019['SO2'],
                    mode='lines+markers',
                    name='SO2 levels of delhi from 23-march-2019  to 15-april 2019'))
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2020['SO2'],
                    mode='lines+markers',
                    name='SO2 levels of delhi from 23-march-2020  to 15-april 2020'))
fig.show()


# In[38]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=delhi_data['Date'], y=delhi_data['Benzene'],
                    mode='lines',
                    name='Benzene level'))
fig.add_trace(go.Scatter(x=delhi_data['Date'], y=delhi_data['Toluene'],
                    mode='lines',
                    name='Toluene level'))
fig.add_trace(go.Scatter(x=delhi_data['Date'], y=delhi_data['Xylene'],
                    mode='lines',
                    name='Xylene level'))
fig.show()


# In[39]:


data_2019=delhi_data[(delhi_data['Date'] >= '2019') & (delhi_data['Date'] < '2020')]
data_2019['BTX']=data_2019['Benzene'] + data_2019['Toluene'] + data_2019['Xylene']


# In[40]:


fig = px.line(data_2019, x="Date", y="BTX")
fig.show()


# In[41]:


fig = px.line(delhi_data, x="Date", y="AQI")
fig.show()


# In[42]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2019['AQI'],
                    mode='lines+markers',
                    name='AQI levels of delhi from 23-march-2019  to 15-april 2019'))
fig.add_trace(go.Scatter(x=march24_2019['Date'], y=march24_2020['AQI'],
                    mode='lines+markers',
                    name='AQI levels of delhi from 23-march-2020  to 15-april 2020'))
fig.show()


# In[43]:


march24_2019['AQI'].mean()


# In[44]:


march24_2020['AQI'].mean()


# In[ ]:




