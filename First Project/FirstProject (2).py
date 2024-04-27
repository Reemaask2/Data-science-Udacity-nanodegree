#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Before starting to import the libraries, I installed some of them directly to the Python environment using command prompt
# I'm using this code to ensure that the installations were successful 
get_ipython().system('pip install plotly')
get_ipython().system('pip install Scipy')


# In[ ]:


#Preparing data


# In[3]:


# we will start by importing necessary libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from scipy import optimize
from scipy import interpolate
from scipy import signal
from IPython import display 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Read the dataset 'Obesity levels Data.csv' into a pandas DataFrame called df 
# Display the first few rows of the dataset to ensure the data structure and its content 
df = pd.read_csv('Obesity levels Data.csv')
df.head()


# In[ ]:


# Assess the dataset 


# In[4]:


# Count the number of rows and columns 
num_rows, num_cols = df.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_cols)


# In[5]:


# Here to retrieve data columns in the dataset
df.columns


# In[6]:


# Here to look at the rows of th dataset
df


# In[7]:


# Retrieve the data types of each column in the dataset
df.dtypes


# In[8]:


# info about datatypes and nulls # it seems that there are no missing values :) 
df.info()


# In[9]:


#Here we see in the "Age" column, the mean age is aprroximately 24 years, and the minimum age observed is 14 years, and the maximum age is 61 years
df.describe()


# In[10]:


sum(df.duplicated())


# In[11]:


# Check coloumns that contain duplicated values and print the columns names 
duplicated_columns = df.columns[df.apply(lambda x: x.duplicated()).any()]

# Print the columns with duplicated values
print("Columns with duplicated values:", list(duplicated_columns))


# In[12]:


# Count the number of duplicated values in the 'Age' column
df.duplicated(subset=['Age']).sum()


# In[13]:


df.Age.value_counts()


# In[14]:


#Cleaning the data #qualit: there are columns with duplicated values 'age, gender'


# In[15]:


# We will make a copy of the dataset, in case we want to go back to the original one
df_cleaned = df.copy()


# In[16]:


# Identify duplicated rows based on the 'Age' attribute
duplicate_age_rows = df[df.duplicated(subset=['Age'], keep='first')]

# Print the duplicated rows for inspection
print("Duplicated rows based on 'Age':")
print(duplicate_age_rows)


# In[ ]:


#Cleaning process ##quality issue: As we see above have floating-point in Values of Age, so we normalize the values


# In[18]:


df['Age'].round().astype(int)

# Print the first few rows to verify the transformation
print(df.head())

# Optionally, save the modified DataFrame to a new CSV file
df.to_csv('normalized_dataset.csv', index=False)


# In[ ]:


#Exploratory Data Analysis: Analyse and Visualise


# In[19]:


#I'm first interested in show the overall Obesity levels ditribution 

obesity_level_counts = df['NObeyesdad'].value_counts()

# Find the tallest bar 
max_count_index = obesity_level_counts.idxmax()

# To customize colors of the bar
custom_colors = ['blue' if idx != max_count_index else 'red' for idx in obesity_level_counts.index]

plt.figure(figsize=(8, 6))
obesity_level_counts.plot(kind='bar', color=custom_colors)

# Show each bar with its corresponding value
for i, count in enumerate(obesity_level_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.title('Distribution of Obesity Levels')
plt.xlabel('Obesity Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#As shown above that Class 2 of obesity level has shown the highest number, based on Google search class 2 of obesity means that IBM of 30 to<35 link: 


# In[ ]:


#What is the Obesity levels distribution over Age Groups?


# In[22]:


Age_obesity_counts = df.groupby(['Age', 'NObeyesdad']).size().unstack()

# Plot a stacked bar chart to visualize the distribution of obesity levels per age
plt.figure(figsize=(12, 8))
Age_obesity_counts.plot(kind='bar', stacked=True)
plt.title('Distribution of Obesity Levels per Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Obesity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:


# The output above is not as expected, i think because we didn't pre-define the Age Groups first 


# In[44]:


Age_bins = [0, 10, 20, 30, 40, 50, 60]
Age_labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60+']

# Categorize ages into age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=Age_bins, labels=Age_labels, right=False)

# Group the data by age group and obesity level, and calculate the count of each group
age_obesity_counts = df.groupby(['AgeGroup', 'NObeyesdad']).size().unstack()

# Plot a stacked bar chart to visualize the distribution of obesity levels per age group
plt.figure(figsize=(12, 8))
age_obesity_counts.plot(kind='bar', stacked=True)
plt.title('Distribution of Obesity Levels per Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Obesity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:


#Above it doesn't show Age group (0-20) which means dropping huge number of values! especially the minimum value of Age column is 14 years old as shown in the preparing data phase


# In[54]:


Age_bins =  [0, 20, 30, 40, 50, 60, 100]
Age_labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60+']

# Categorize ages into age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=Age_bins, labels=Age_labels, right=False)

# Group the data by age group and obesity level, and calculate the count of each group
Age_group_obesity_counts = df.groupby(['AgeGroup', 'NObeyesdad']).size().unstack()

# Show a stacked bar chart to visualize the distribution of obesity levels for each age group
plt.figure(figsize=(12, 8))
Age_group_obesity_counts.plot(kind='bar', stacked=True)
plt.title('Distribution of Obesity Levels per Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Obesity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[49]:


Age_group = '20-30'  
obesity_counts = Age_group_obesity_counts.loc[age_group].dropna()
plt.figure(figsize=(8, 8))
plt.pie(obesity_counts, labels=obesity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title(f'Distribution of Obesity Levels for Age Group {Age_group}')
plt.axis('equal') 
plt.show()


# In[58]:


Age_group = '30-40'  # Example age group (you can choose any age group from your dataset)
obesity_counts = age_obesity_counts.loc[age_group].dropna()  # Corrected variable name
plt.figure(figsize=(8, 8))
plt.pie(obesity_counts, labels=obesity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title(f'Distribution of Obesity Levels for Age Group {Age_group}')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[ ]:


#Is there a relationship between frequent consumption of high caloric food and obesity?
#I'm insterested in show and determine some of the potential factors of obesity levels 


# In[11]:


caloric_obesity_counts = df.groupby(['FAVC', 'NObeyesdad']).size().unstack()

# Show a grouped bar chart
caloric_obesity_counts.plot(kind='bar', figsize=(10, 6))
plt.title('Distribution of Obesity Levels by high caloric food')
plt.xlabel('Physical Activity Level')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Obesity Level')
plt.show()


# In[ ]:


# We can see above that people who answer with "Yes" are suffering of different obesity levels, what is really insteresting that there are large number of people suffering from insufficient weight even if they highly consumpt high caloric food which let me ask myself this question: Are they consumpt high caloric food to get normal weight?  


# In[ ]:


# Is there a relationship between the time spent using technological devices and obesity levels?


# In[17]:


plt.figure(figsize=(10, 6))
plt.scatter(df['TUE'], df['NObeyesdad'], alpha=0.7, c='blue', s=50)
plt.title('Relationship between Time Spent Using Technological Devices and Obesity Levels')
plt.xlabel('Time Spent Using Technological Devices')
plt.ylabel('Obesity Level')
plt.grid(True)
plt.show()


# In[ ]:


#Worth mention that based on a quick research for "What does TUE column values could mean?" it appears that they provide a way to categorize individuals based on their level of engagement with technology devices 


# In[19]:


#Is there a relationship between family history of overweight and obesity levels?


# In[21]:


# Group the data by family history of overweight and obesity levels
family_obesity_counts = df.groupby(['family_history_with_overweight', 'NObeyesdad']).size().unstack()

# Plot a grouped bar chart
family_obesity_counts.plot(kind='bar', figsize=(10, 6))
plt.title('Distribution of Obesity Levels by Family History of Overweight')
plt.xlabel('Family History of Overweight')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Obesity Level')
plt.show()


# In[ ]:


#It is very obvious that there is a relationship between Family History of Overweight and Obesity levels! Especially with obesity class 1 has got the hight value! 


# In[ ]:


#Conclusion 


# In[ ]:


# 1. Higher Rate in Families with a History of Overweight: Compared to families without such a history, families with a history of overweight often have a higher rate of obesity.  
# 2. Influence of Family Environment: The presence of family members who are overweight, it may be an indicator of common lifestyle choices or environmental factors—such as dietary habits, physical activity patterns, or even socioeconomic status—that cause obesity in the family. 
# 3.Awareness  Opportunities: By identifying the correlation between obesity levels and family histories of overweight will healthcare professionals provide targeted interventions that reduce the chance of obesity in high-risk families.
# 4. The Importance of Family-Based Interventions: Given the potential influence of family history on obesity risk, interventions aimed at the entire family may be more effective in promoting healthy lifestyle behaviors and preventing obesity than individual-focused approaches. Family-based interventions may include education, behavioral counseling, and support for adopting healthier habits together.
#5.Need for Additional Research: While this analysis provides useful insights, more research is needed to understand the complex interplay between genes, environmental factors, and the risk of obesity within families.
#6. Individuals who spend more time using technological (smart) devices may be more likely to suffer from obesity in the future, especially if they do not follow a healthy lifestyle. 
# 7.There is a strong correlation between time spent on technological devices and insufficient weight! which I recommend doing my health research to determine if there is a physiological or other factor leading to it. 
# 8 Obesity_Type_II is the dominant obesity level for people aged 20-30, whereas Obesity_Type_I is the dominant obesity level for people aged 30-40. What factors have changed? Is it true that people become more conscious of their health as they get older? 

