
# coding: utf-8

# In[4]:

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity as cosine
from scipy.sparse.csgraph import connected_components
import itertools
import json
import numpy as np
import glob
from dsnmf import DSNMF, appr_seminmf
n_topics = 20
n_top_words = 20
###fake data
# dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                              remove=('headers', 'footers', 'quotes'))
# data_samples = dataset.data


# In[5]:

local_data = []
classes = ['afghannationalliberationfront', 'hezbislami', 'jamiatislami']
philes =  glob.glob("/Users/ziv/GDrive/school/math-thesis/nmf-imp/txt_data_bypage/*.txt")
Y = np.zeros((len(classes),len(philes)))
for (i,phile) in enumerate(philes):
    c = phile.split('/')[-1].split('_')[0]
    cls = classes.index(c)
    Y[cls,i]= 1
    with open(phile, 'r') as myfile:
        data=myfile.read().replace('\n', '')
        local_data.append(unicode(data, errors='ignore'))


# In[6]:

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')

X = tfidf_vectorizer.fit_transform(local_data)


# In[7]:

tfidf_feature_names = tfidf_vectorizer.get_feature_names()
dsnmf = DSNMF(X.toarray(), layers=(400, 100))
for epoch in range(100):
    residual = float(dsnmf.train_fun())
    
    print("Epoch {}. Residual [{:.2f}]".format(epoch, residual))
fea = dsnmf.get_features().T # this is the last layers features i.e. h_2


# In[16]:

a1 = dsnmf.get_param_values()[0]
a2 = dsnmf.get_param_values()[1]
s = dsnmf.get_param_values()[2]


# In[19]:

print X.shape
print a1.shape
print a2.shape
print s.shape


# In[ ]:




# In[8]:

H = nmf.components_
W = nmf.fit_transform(tfidf)
#x = WH
weights = (5000/W.sum())*W.sum(axis=0)

def build_wgraph(alpha=2):
    if alpha != 2:
        return [[int(cosine(H[i],H[j])[0][0] > alpha) for i in range(0, len(H))] for j in range(0, len(H))]
    else:
        return [[cosine(H[i],H[j])[0][0] for i in range(0, len(H))] for j in range(0, len(H))]
def thresh_vals(numbin):
    binz = []
    w = build_wgraph(2)
    chain = itertools.chain(*w)
    s =sorted(list(chain))
    val = n_topics*n_topics/numbin
    for i,v in enumerate(s):
        if i%val ==0: binz.append(v)
    return binz


# In[13]:

def array_distance(A,B):
    count = 0
    for i,x in enumerate(A):
        if x == B[i]:
            count+=1
    return len(A)-count

def greedy_TV_build(to_consume,bins):
    if len(to_consume)>1:
        a = to_consume[0]
        b = to_consume[1]
        ccA = connected_components(build_wgraph(a))[1]
        ccB = connected_components(build_wgraph(b))[1]
        if not np.array_equal(ccA, ccB):
            distance = array_distance(ccA,ccB)
            if distance > 8:
                new_tv = [a + i*(b-a)/bins for i in range(0,bins)]
                return new_tv + greedy_TV_build( to_consume[1:], bins)
            else:
                return [a] + greedy_TV_build(to_consume[1:], bins)
        else:
            return greedy_TV_build(to_consume[1:], bins)
    elif len(to_consume) == 1:
        return to_consume
    else:
        return []
  

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
                indices = [i for i, x in enumerate(parent_community) if x == unique_community] #[8,9]
                seen_communities = []
                for i in indices: #8 and 9
                    if child_community[i] in seen_communities:
                        filter(lambda x: x['community'] == str(child_community[i]), children)[0]['indices'].append(i)
                    else:
                        community_to_find = child_community[i]
                        grow_my_children = populateTree(row_level+1, community_to_find)
                        if grow_my_children:
                            name = ""
                            children.append({"community":str(child_community[i]),"indices":[i],"name" : name , "children":grow_my_children, "hasChildren": True})
                        else:
                            name = " ".join([tfidf_feature_names[j] for j in nmf.components_[i].argsort()[:-n_top_words - 1:-1]])
                            children.append({"community":str(child_community[i]),"indices":[i],"size":weights[i],"name":name, "hasChildren": False})
                        seen_communities.append(child_community[i])
        if len(children) == 1:
            try: 
                return children[0]['children']
            except:
                return []
        else:
            return children
        
def recursiveNaming(tree):
        i = tree['indices']
        base = nmf.components_[i[0]]
        if len(i) > 1:
            for ind in i[1:]:
                base = np.add(nmf.components_[ind], base)
        tree['name'] = " ".join([tfidf_feature_names[j] for j in base.argsort()[:-n_top_words - 1:-1]])
        if tree['hasChildren']:
            for child in tree['children']:
                    recursiveNaming(child)


# In[ ]:




# In[14]:

tv = greedy_TV_build(thresh_vals(100),2)
# visualize topic tree
# for i in tv:
#     ccB = connected_components(build_wgraph(i))[1]
#     print ccB
size = len(tv)


# In[15]:

flare = {"name" : "" , "children" : populateTree(0, 0)}
for child in flare['children']:
    recursiveNaming(child)
with open('demo.json', 'w') as outfile:
    json.dump(flare, outfile)


# In[33]:

tv


# In[ ]:



