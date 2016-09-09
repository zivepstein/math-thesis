
# coding: utf-8

# In[2]:

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity as cosine
import itertools
import json


# In[3]:


n_topics = 10
n_top_words = 20
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data


# In[64]:

print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(local_data)


nmf = NMF(n_components=n_topics).fit(tfidf)


# In[62]:

local_data = []
philes =  glob.glob("/Users/ziv/GDrive/school/thesis/nmf-imp/data/*.txt")
for phile in philes:
    print phile
    with open(phile, 'r') as myfile:
        data=myfile.read().replace('\n', '')
        local_data.append(data)


# In[5]:

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)


# In[63]:

local_data


# In[6]:

H = nmf.components_
def build_wgraph(alpha=2):
    if alpha != 2:
        return [[int(cosine(H[i],H[j])[0][0] > alpha) for i in range(0, len(H))] for j in range(0, len(H))]
    else:
        return [[cosine(H[i],H[j])[0][0] for i in range(0, len(H))] for j in range(0, len(H))]


# In[7]:

def thresh_vals(numbin):
    binz = []
    w = build_wgraph(2)
    chain = itertools.chain(*w)
    s =sorted(list(chain))
    val = n_topics*n_topics/numbin
    for i,v in enumerate(s):
        if i%val ==0: binz.append(v)
    return binz


# In[8]:

def compare_graphs(graphA, graphB):
    return 47


# In[9]:

seed = build_wgraph(0)


# In[12]:

for t in thresh_vals(10):
    graphB = build_wgraph(t)
    compare_graphs(seed, graphB)
    seed = graphB


# In[13]:

from scipy.sparse.csgraph import connected_components


# In[14]:

for i in thresh_vals(10):
    cc = connected_components(build_wgraph(i))
    print cc[1]


# In[165]:

cc1 = [0,0,0,0,0,1,1,0,1,1]
cc2 = [0, 0 ,0 ,1, 2, 3, 4 ,0 ,5, 5]


# In[15]:




# In[46]:

size = 10
tv = thresh_vals(size)
cc1 = connected_components(build_wgraph(tv[0]))[1]
cc2 = connected_components(build_wgraph(tv[1]))[1]


def populateTree(row_level, valid_community):
    if row_level > size-2 :
        return []
    else:
        children = []
        parent_community = connected_components(build_wgraph(tv[row_level]))[1]
        child_community = connected_components(build_wgraph(tv[row_level+1]))[1]
        unique_communities = list(set(parent_community))
        for unique_community in unique_communities:
            if valid_community == unique_community:
                indices = [i for i, x in enumerate(parent_community) if x == unique_community]
                seen_communities = []
                for i in indices:
                    if child_community[i] in seen_communities:
                        filter(lambda x: x['community'] == str(child_community[i]), children)[0]['indices'].append(i)
                    else:
                        community_to_find = child_community[i]
                        grow_my_children = populateTree(row_level+1, community_to_find)
                        if grow_my_children:
                            children.append({"community":str(child_community[i]),"indices":[i],"name" : "" , "children":grow_my_children})
                        else:
                            name = " ".join([tfidf_feature_names[j] for j in nmf.components_[i].argsort()[:-n_top_words - 1:-1]])
                            children.append({"community":str(child_community[i]),"indices":[i],"size":500,"name":name})
                        seen_communities.append(cc2[i])
        return children
    
flare = {"name" : "flare" , "children" : populateTree(0, 0)}

with open('demo.json', 'w') as outfile:
    json.dump(flare, outfile)


# In[44]:

name = "<br>".join([tfidf_feature_names[i] for i in nmf.components_[i].argsort()[:-n_top_words - 1:-1]])


# In[39]:

name


# In[ ]:



