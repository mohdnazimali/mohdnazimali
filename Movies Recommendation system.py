#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv("bollywood_full.csv")


# In[4]:


data.head()


# In[5]:


for i in data.columns:
    print(i)


# In[6]:


data=data.drop(["year_of_release", "tagline","release_date","wins_nominations", "title_x","imdb_id", "poster_path", "wiki_link", "title_y", "runtime", "imdb_rating","imdb_votes"],axis=1)


# In[7]:


for i in data.columns:
    print(i)


# In[8]:


data.head()


# In[9]:


data["is_adult"].value_counts()


# In[10]:


data=data.drop(["is_adult", "story"],axis=1)


# In[11]:


data.head()


# In[12]:


data.isna().sum()


# In[13]:


data=data[data["actors"].notna()]


# In[14]:


data.isna().sum()


# In[15]:


data.head(1)["actors"].value_counts()


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer 


# In[17]:


tfidf=TfidfVectorizer(min_df=3,ngram_range=(1,3),stop_words="english",analyzer="word")


# In[18]:


tf_matrix=tfidf.fit_transform(data["actors"])


# In[19]:


print(tf_matrix.shape)


# In[20]:


from sklearn.metrics.pairwise import sigmoid_kernel 


# In[21]:


sgm=sigmoid_kernel(tf_matrix,tf_matrix)


# In[22]:


sgm[0]


# In[23]:


indicse=pd.Series(data.index,index=data["original_title"]).drop_duplicates()


# In[25]:


def recommdetion(name,sgm=sgm):
    idx= indicse[name]
    data_s=list(enumerate(sgm[idx]))
    data_s=sorted(data_s,key = lambda x:x[1],reverse=True )
    top_10= data_s[1:11]
    data_indese=[i[0]for i in top_10]
    return data["original_title"].iloc[data_indese]


# In[26]:


recommdetion("Uri: The Surgical Strike")


# In[ ]:




