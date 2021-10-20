#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris_dataset = load_iris()


# In[3]:


print("Keys of iris_dataset:\n", iris_dataset.keys())


# In[4]:


print(iris_dataset['DESCR'][:193] + "\n...")


# In[5]:


print("Target names:", iris_dataset['target_names'])


# In[7]:


print("Feature names:\n", iris_dataset['feature_names'])


# In[8]:


print("Type of data:", type(iris_dataset['data']))


# In[9]:


print("Shape of data:", iris_dataset['data'].shape)


# In[10]:


print("First five rows of data:\n", iris_dataset['data'][:5])


# In[11]:


print("Type of target:", type(iris_dataset['target']))


# In[12]:


print("Shape of target:", iris_dataset['target'].shape)


# In[13]:


print("Target:\n", iris_dataset['target'])


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[15]:


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# In[16]:


print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[25]:


import pandas as pd 
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8)


# In[ ]:





# In[ ]:




