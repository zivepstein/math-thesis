
# coding: utf-8

# In[1]:

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity as cosine
from scipy.sparse.csgraph import connected_components
import itertools
import json
import numpy as np
import glob
n_topics = 20
n_top_words = 20
###fake data
# dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                              remove=('headers', 'footers', 'quotes'))
# data_samples = dataset.data


# In[54]:

# In[2]:

##generate data as array of strings from local .txt files
local_data = []
timetable = []
philes =  glob.glob("/Users/ziv/GDrive/school/math-thesis/nmf-imp/txt_data_bypage/*.txt")
for phile in philes:
    s = phile.split('/')[-1]
    result = re.search('%s(.*)%s' % ('_', '_'), s).group(1)
    if len(result) == 5: 
        time = datetime.datetime(int(result[0:4]), int(result[4]), 1)
    else:
        time =    datetime.datetime(int(result[0:4]), int(result[4:6]), 1)
    timetable.append(time)
    with open(phile, 'r') as myfile:
        data=myfile.read().replace('\n', '')
        local_data.append(unicode(data, errors='ignore'))


# In[3]:


# In[56]:

sorted(timetable)


# In[36]:

len(local_data)


# In[4]:

#tfdif and nmf model building
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(local_data)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
nmf = NMF(n_components=n_topics).fit(tfidf)


# In[5]:

H = nmf.components_
W = nmf.fit_transform(tfidf)


# In[39]:

W.shape


# In[40]:

len(timetable)


# In[58]:

len(sorted(list(set(timetable))))


# In[53]:

import datetime


# In[61]:

(sorted(timetable))


# In[ ]:



