
# coding: utf-8

# In[2]:

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity as cosine
from scipy.sparse.csgraph import connected_components
import itertools
import json
import numpy as np
import glob


# In[3]:

##generate data as array of strings from local .txt files
local_data = []
philes =  glob.glob("/Users/ziv/GDrive/school/math-thesis/nmf-imp/data/*.txt")
for phile in philes:
    with open(phile, 'r') as myfile:
        data=myfile.read().replace('\n', '')
        local_data.append(unicode(data, errors='ignore'))


# In[4]:

#tfdif and nmf model building
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')
    
tfidf = tfidf_vectorizer.fit_transform(local_data)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


# In[54]:

def generate_graphs(X,p):
    W_raw = np.zeros((X.shape[1],X.shape[1]))
    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[0]):
            W_raw[i,j] = np.dot(X[i,], t(X[j,]))[0,0]
    s = W_raw.flatten()
    thresh = np.percentile(s,p)
    W = (W_raw > thresh).astype(int)
    D = np.zeros((X.shape[1],X.shape[1]))
    for i in range(0,X.shape[0]):
        D[i,i] = sum(W[i,])
    return W,D


# In[83]:

def grnmf(X,k,lam,p,u0=None,v0=None):
    W,D = generate_graphs(X,p)
    if u0==None: 
        u = np.random.rand(X.shape[0],k)
    else:
        u = u0
    if v0==None:
        v = np.random.rand(X.shape[1],k)
    else:
        v = v0
    i = 0    
    while i < 1000:
        u = u = np.multiply(u,((X*v)/ np.dot(np.dot(u,t(v)),v)))
        v = np.multiply(v , ((np.dot(t(X),u) + 0.2*np.dot(W,v))/ (np.dot(np.dot(v,t(u)),u) + .2*np.dot(D,v))))
        i +=1
        if i % 5 ==0:
            print str(i) + "/1000 iterations complete"
    return u,v


# In[84]:

def t(x):
    return np.transpose(x)


# In[87]:

u,v = grnmf(X,k=10,lam=0.2,p=90)


# In[ ]:



