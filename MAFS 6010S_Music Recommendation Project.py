# coding: utf-8

# # MAFS 6010S: Music Recommendation Project

# In[17]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import cross_validation, grid_search, metrics, ensemble


# In[18]:


train = pd.read_csv('train.csv')
train = train.sample(frac = 0.03)
songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv')
data_1 = pd.merge(train, songs, on = 'song_id', how = 'left')
data = pd.merge(data_1, members, on = 'msno', how = 'inner')


# In[19]:


data


# In[20]:


for i in data.select_dtypes(include = ['object']).columns:
    data[i][data[i].isnull()] = 'unknown'
data = data.fillna(value = 0)


# In[21]:


plt.figure(figsize = (8,6))
sns.countplot(data['source_system_tab'], hue = data['target'])


# In[22]:


plt.figure(figsize = (8,6))
source_type_plot = sns.countplot(data['source_type'], hue = data['target'])
source_type_plot.set_xticklabels(labels,rotation = 90)


# In[23]:


# Deal with registration_init_time and expiration_date 
data.expiration_date = pd.to_datetime(data.expiration_date, format = '%Y%m%d', errors = 'ignore')
data.registration_init_time = pd.to_datetime(data.registration_init_time, format = '%Y%m%d', errors = 'ignore')
data['expiration_year'] = data['expiration_date'].dt.year
data['expiration_month'] = data['expiration_date'].dt.month
data['expiration_day'] = data['expiration_date'].dt.day
data['registration_year'] = data['registration_init_time'].dt.year
data['registration_month'] = data['registration_init_time'].dt.month
data['registration_day'] = data['registration_init_time'].dt.day
data['expiration_date'] = data['expiration_date'].astype('category')
data['registration_init_time'] = data['registration_init_time'].astype('category')
for temp in data.select_dtypes(include = ['object']).columns:
    data[temp] = data[temp].astype('category') 
for temp in data.select_dtypes(include = ['category']).columns:
    data[temp] = data[temp].cat.codes


# In[24]:


data.info()


# In[25]:


# Correlation matrix of factors
sns.heatmap(data.corr())
plt.show()


# In[26]:


data_all = data.copy(deep = True)
data_all_1 = data_all.pop('target')
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data_all, data_all_1, test_size = 0.4)


# In[27]:


# Random forest model
forest_model = ensemble.RandomForestClassifier(n_estimators = 300, max_depth = 30)
forest_model.fit(train_data, train_labels)
predict_labels = forest_model.predict(test_data)
print(metrics.classification_report(test_labels, predict_labels))


# In[28]:


# Factor contributuion
factor_plot = pd.DataFrame({'factor': data_all.columns[data_all.columns != 'target'], 'contribution': forest_model.feature_importances_})
factor_plot = factor_plot.sort_values('contribution', ascending=False)
sns.barplot(x = factor_plot.contribution, y = factor_plot.factor)
plt.show()


# In[29]:


# Select foctors with contribution > 0.04 (above registration_month)
data_all_improve = data.copy(deep = True)
data_all_improve = data_all_improve[['msno', 'song_id', 'song_length', 'source_type', 'artist_name', 'target', 'composer', 'expiration_day', 
                                     'registration_init_time', 'registration_month', 'registration_day']]


# In[30]:


# Correlation matrix of remaining factors
sns.heatmap(data_all_improve.corr())
plt.show()


# In[31]:


data_all_improve_1 = data_all_improve.pop('target')
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data_all_improve, data_all_improve_1, test_size = 0.4)


# In[32]:


# Random forest model
forest_model_improve = ensemble.RandomForestClassifier(n_estimators = 300, max_depth = 30)
forest_model_improve.fit(train_data, train_labels)
predict_labels_improve = forest_model_improve.predict(test_data)
print(metrics.classification_report(test_labels, predict_labels_improve))