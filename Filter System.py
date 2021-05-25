#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings; warnings.simplefilter('ignore')


# In[2]:


df = pd.read_csv('IMDb movies.csv')
dfn = pd.read_csv('IMDb names.csv')
dfr = pd.read_csv('IMDb ratings.csv')
dft = pd.read_csv('IMDb title_principals.csv')


# In[3]:


df.info()


# In[5]:


dfn.head()


# In[6]:


df.isnull().sum() # verificando se tinha algum valor NaN no avg_test ou votes


# In[7]:


colunas = ['original_title', 'year', 'genre','country','director','actors', 'avg_vote','votes']
dfnovo = df[colunas]


# In[8]:


dfnovo.head()


# In[9]:


votes = dfnovo[dfnovo["votes"].notnull()]["votes"].astype('int')


# In[10]:


dfnovo.info()


# In[11]:


avg_vote = dfnovo[dfnovo["avg_vote"].notnull()]["avg_vote"].astype('float')


# In[12]:


dfnovo.director.fillna("Nenhum diretor", inplace = True)
dfnovo.actors.fillna(" Nenhum Ator", inplace = True)


# In[13]:


C = avg_vote.mean()
C


# In[22]:


#calculando o quartil para entrar no top filmes
m = votes.quantile(0.40)
m


# In[23]:


qualified = dfnovo[(dfnovo['votes'] >= m) & (dfnovo['votes'].notnull()) & (dfnovo['avg_vote'].notnull())]


# In[24]:


qualified.info()


# In[25]:


qualified.shape


# In[22]:


def imdb_qualified(x):
    v = x["votes"]
    R = x["avg_vote"]
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[23]:


qualified['wr'] = qualified.apply(imdb_qualified, axis = 1)


# In[24]:


qualified.info()


# In[25]:


qualified = qualified.sort_values("wr", ascending = False).head(250)


# In[26]:


qualified.head(20)


# In[28]:


#Sistema de recomendação 2


# In[ ]:


#Função de top filmes por categoria 

