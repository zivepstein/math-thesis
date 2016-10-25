
# coding: utf-8

# In[190]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pylab as plt


# In[215]:

def nmf(A,k,u0=None,v0=None):
    if u0==None: 
        u = np.random.rand(A.shape[0],k)
    else:
        u = u0
    if v0==None:
        v = np.random.rand(k,A.shape[1])
    else:
        v = v0
    i = 0    
    while i < 1000:
        u = u * (np.dot(A,t(v))/ np.dot(np.dot(u,v),t(v)))
        v = v * t(np.dot(t(A),u)/ t(np.dot(np.dot(t(u), u), v)))
        i +=1
    return u,v
def t(x):
    return np.transpose(x)


# In[216]:

words = 100
docs_per_word = 5
bin_size = 100
noise_parameter = 1
docs = words * docs_per_words
A = np.zeros((words,docs))
word_bin_size = bin_size/docs_per_word
s = np.zeros((word_bin_size,bin_size))
s.fill(1)


# In[217]:

for i in range(0,docs/bin_size):
    A[i*word_bin_size:(i+1)*word_bin_size,i*bin_size:(i+1)*bin_size] = s


# In[218]:

A +=np.random.rand(A.shape[0],A.shape[1])*noise_parameter


# In[219]:

(u,v) = nmf(A,10)


# In[221]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(A, interpolation='nearest', cmap=plt.cm.ocean)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(u, interpolation='nearest', cmap=plt.cm.ocean)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(v, interpolation='nearest', cmap=plt.cm.ocean)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(np.dot(u,v), interpolation='nearest', cmap=plt.cm.ocean)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



