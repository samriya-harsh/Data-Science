#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.style.use("Solarize_Light2")
#sns.set_palette("dark")
#sns.set_style("ticks")

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[2]:


bigbasket = pd.read_csv("BigBasket Products.csv.zip")
bigbasket.head(20)


# In[3]:


bigbasket.describe()


# In[11]:


bigbasket.info()


# In[12]:


bigbasket.isnull()


# In[14]:


bigbasket.shape


# In[15]:


bigbasket.tail(8)


# In[16]:


top= bigbasket["product"].value_counts().head(15)
least=bigbasket["product"].value_counts().tail(15)


# In[17]:


fig= plt.figure(figsize=(14,4))
ax = fig.add_axes([0,0,1,1])

import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)

    


sns.barplot(x=top.index, y=top.values, data=bigbasket,
            linewidth=0,
            alpha=1.0,
            color="#4b94cf")

#format axis
ax.set_xlabel("Most sold products",fontsize=15, weight='semibold')
ax.set_ylabel("Number",fontsize=15, weight='semibold')

wrap_labels(ax, 10)

plt.show()


# In[18]:


fig= plt.figure(figsize=(14,4))
ax = fig.add_axes([0,0,1,1])

sns.barplot(x=least.index, y=least.values, data=bigbasket, linewidth=0, alpha=1.0, color="darkgoldenrod")
sns.set(style="ticks")

#format axis
ax.set_xlabel("Least sold products",fontsize=15, weight='semibold')
ax.set_ylabel("Number",fontsize=15, weight='semibold')
plt.xticks(fontsize=10, weight="semibold")


wrap_labels(ax, 10)
fig.show()


# In[19]:


fig= plt.figure(figsize=(14,4))
ax2 = fig.add_axes([0,0,1,1])

sns.barplot(x=bigbasket["category"].value_counts()[:10].index, y=bigbasket["category"].value_counts()[:10].values,
            data=bigbasket, linewidth=0,
            alpha=1.0,
            color="b")

wrap_labels(ax2,10)


# In[20]:


bigbasket["diff_in_prices"] = bigbasket["market_price"] - bigbasket["sale_price"]
#bigbasket.head()
discount = bigbasket[bigbasket["diff_in_prices"] != 0]
discount


# In[21]:


fig = plt.figure(figsize=(20,10))

plt.style.use("Solarize_Light2")
sns.distplot(discount.rating, color='b', kde =True)
sns.distplot(bigbasket.rating, color='gold', kde =True)
plt.xlabel("Ratings",fontsize=15, weight='semibold')
plt.ylabel("Density",fontsize=15, weight='semibold')
plt.title("Relative distribution of all products with discounted products",fontsize=15, weight='semibold')
fig.legend()


# In[25]:


fig= plt.figure(figsize=(14,4))
ax2 = fig.add_axes([0,0,1,1])

sns.barplot(x=bigbasket["category"].value_counts()[:10].index, y=bigbasket["category"].value_counts()[:10].values,
           data=bigbasket, linewidth=0,
           alpha=1.0,
           color="b")

wrap_labels(ax2,10)


# In[26]:


counts = bigbasket['sub_category'].value_counts()

counts_df_1 = pd.DataFrame({'Category':counts.index,'Counts':counts.values})[:10]


# In[27]:


px.bar(data_frame=counts_df_1,
 x='Category',
 y='Counts',
 color='Counts',
 color_continuous_scale='blues',
 text_auto=True,
 title=f'Top 10 Bought Sub_Categories')


# In[29]:


counts = bigbasket['brand'].value_counts()

counts_bigbasket_brand = pd.DataFrame({'Brand Name':counts.index,'Counts':counts.values})[:10]


# In[30]:


px.bar(data_frame=counts_bigbasket_brand,
 x='Brand Name',
 y='Counts',
 color='Counts',
 color_continuous_scale='blues',
 text_auto=True,
 title=f'Top 10 Brand Items based on Item Counts')


# In[32]:


counts = bigbasket['type'].value_counts()

counts_bigbasket_type = pd.DataFrame({'Type':counts.index,'Counts':counts.values})[:10]


# In[34]:


px.bar(data_frame=counts_bigbasket_type,
 x='Type',
 y='Counts',
 color='Counts',
 color_continuous_scale='blues',
 text_auto=True,
 title=f'Top 10 Types of Products based on Item Counts')


# In[38]:


def sort_recommendor(col='rating',sort_type = False):
    """
    A recommendor based on sorting products on the column passed.
    Arguments to be passed:
    
    col: The Feature to be used for recommendation.
    sort_type: True for Ascending Order
    """
    rated_recommend = bigbasket.copy()
    if rated_recommend[col].dtype == 'O':
        col='rating'
    rated_recommend = rated_recommend.sort_values(by=col,ascending = sort_type)
    return rated_recommend[['product','brand','sale_price','rating']].head(10)


# In[39]:


help(sort_recommendor)


# In[40]:


sort_recommendor(col='sale_price',sort_type=True)


# In[48]:


C= bigbasket['rating'].mean()
C


# In[52]:


def sort_recommendor(col='rating',sort_type = False):
    """
    A recommendor based on sorting products on the column passed.
    Arguments to be passed:
    
    col: The Feature to be used for recommendation.
    sort_type: True for Ascending Order
    """
    rated_recommend = bigbasket.copy().loc[bigbasket['rating'] >= 3.5]
    if rated_recommend[col].dtype == 'O':
        col='rating'
    rated_recommend = rated_recommend.sort_values(by=col,ascending = sort_type)
    return rated_recommend[['product','brand','sale_price','rating']].head(10)


# In[53]:


sort_recommendor(col='sale_price',sort_type=True)


# In[54]:


bigbasket.head()


# In[59]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(bigbasket['description'])
tfidf_matrix.shape


# In[ ]:





# In[ ]:





# In[ ]:




