
# coding: utf-8

# In[35]:

import json
from pprint import pprint

with open('news24.json') as data_file:    
    data = json.load(data_file)


# In[36]:


def find_leaves(x):
    try: 
        for c in x['children']:
            find_leaves(c)
    except:
        print x['name'] + "\n\n\n"


# In[37]:

find_leaves(data)


# In[14]:

data


# In[19]:

for key, value in data.items():
    pprint("Key:")
    pprint(key)


# In[ ]:



