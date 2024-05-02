#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Importing all the libraries
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[11]:


# Read the CSV file into a DataFrame named df 
df=pd.read_csv('CVD_cleaned.csv.zip')


# In[12]:


df.head(10)


# In[13]:


df.shape


# In[14]:


df.info()


# In[15]:


df.isnull().sum()


# In[16]:


df.describe()


# In[17]:


# Count the duplicated rows in the DataFrame
df.duplicated().sum()


# In[18]:


#drop the duplicated values
df.drop_duplicates()


# In[19]:


df.nunique()


# In[20]:


#histogram for BMI 
plt.figure(figsize=(10, 6))
plt.hist(df['BMI'], bins=30, edgecolor='black')
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()


# In[21]:


#histogram for height
plt.figure(figsize=(10, 6))
plt.hist(df['Height_(cm)'], bins=30, edgecolor='black')
plt.title('Distribution of height')
plt.xlabel('height')
plt.ylabel('Frequency')
plt.show()


# In[22]:


#histogram for weight 
plt.figure(figsize=(10, 6))
plt.hist(df['Weight_(kg)'], bins=30, edgecolor='black')
plt.title('Distribution of weight')
plt.xlabel('weight')
plt.ylabel('Frequency')
plt.show()


# In[23]:


# Calculate counts for each category
sex_counts = df['Sex'].value_counts()
age_counts = df['Age_Category'].value_counts()
smoking_counts = df['Smoking_History'].value_counts()
General_Health_counts=df['General_Health'].value_counts()

# Creating subplots for all three bar charts
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 10))

# Bar chart for Sex
axes[0].bar(sex_counts.index, sex_counts.values, color='lightblue')
axes[0].set_title('Distribution of Sex')
axes[0].set_xlabel('Sex')
axes[0].set_ylabel('Count')

# Bar chart for Age_Category
axes[1].bar(age_counts.index, age_counts.values, color='lightgreen')
axes[1].set_title('Distribution of Age Categories')
axes[1].set_xlabel('Age Categories')
axes[1].set_ylabel('Count')

# Bar chart for Smoking_History
axes[2].bar(smoking_counts.index, smoking_counts.values, color='lightcoral')
axes[2].set_title('Smoking History')
axes[2].set_xlabel('Smoking History')
axes[2].set_ylabel('Count')
# Bar chart for general health
axes[3].bar(General_Health_counts.index,General_Health_counts.values,color='yellow')
axes[3].set_title('distribution of general health')
axes[3].set_xlabel('health')
axes[3].set_ylabel('count')
plt.tight_layout()
plt.show()


# In[24]:


# Explore relationships between height and weight 
plt.figure(figsize=(20, 10))
plt.scatter(df['Height_(cm)'],df['Weight_(kg)'])
plt.title('Explore relationships between height and weight')
plt.xlabel('height')
plt.ylabel('weight')
plt.show()


# In[25]:


#boxplot for Sex and BMI
plt.figure(figsize=(12, 10))
sns.boxplot(x='Sex', y='BMI', data=df, palette='coolwarm')
plt.title('Boxplot of BMI by Sex Category')
plt.xlabel('Sex')
plt.ylabel('BMI')
plt.show()


# In[26]:


# Create a copy of the DataFrame to avoid modifying the original
df_encoded = df.copy()

# Create a label encoder object
label_encoder = LabelEncoder()

# Iterate through each object column and encode its values
for column in df_encoded.select_dtypes(include='object'):
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

# Now, df_encoded contains the label-encoded categorical columns
df_encoded.head()


# In[27]:


# Calculate the correlation matrix for Data
correlation_matrix = df_encoded.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[28]:


# Food Consumption Patterns Across Age Categories
heatmap_data = df.pivot_table(index='Age_Category', values=['Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption'])
# Create a heatmap
plt.figure(figsize=(15, 6))
sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".1f", linewidths=.5)
plt.title('Food Consumption Patterns Across Age Categories')
plt.show()

