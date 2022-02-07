#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

data=pd.read_csv("bollywood_full.csv")
data.head()

for i in data.columns:
    print(i)

data=data.drop(["year_of_release", "tagline","release_date","wins_nominations", "title_x","imdb_id", "poster_path", "wiki_link", "title_y", "runtime", "imdb_rating","imdb_votes"],axis=1)

for i in data.columns:
    print(i)
    
data.head()

data["is_adult"].value_counts()

data=data.drop(["is_adult", "story"],axis=1)

data.head()

data.isna().sum()

data=data[data["actors"].notna()]

data.isna().sum()

data.head(1)["actors"].value_counts()

from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf=TfidfVectorizer(min_df=3,ngram_range=(1,3),stop_words="english",analyzer="word")

tf_matrix=tfidf.fit_transform(data["actors"])

print(tf_matrix.shape)

from sklearn.metrics.pairwise import sigmoid_kernel 

sgm=sigmoid_kernel(tf_matrix,tf_matrix)

sgm[0]

indicse=pd.Series(data.index,index=data["original_title"]).drop_duplicates()

def recommdetion(name,sgm=sgm):
    idx= indicse[name]
    data_s=list(enumerate(sgm[idx]))
    data_s=sorted(data_s,key = lambda x:x[1],reverse=True )
    top_10= data_s[1:11]
    data_indese=[i[0]for i in top_10]
    return data["original_title"].iloc[data_indese]

recommdetion("Uri: The Surgical Strike")




