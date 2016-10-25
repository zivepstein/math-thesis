
# coding: utf-8

# In[76]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pylab as plt
import random
import glob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[65]:

def ssnmf(X,L,Y,lamb,r,k,A0=None,S0=None):
    if A0==None: 
        A = np.random.rand(X.shape[0],r)
    else:
        A = A0
    if S0==None:
        S = np.random.rand(r,X.shape[1])
    else:
        S = S0
    B = np.random.rand(k,r)
    i = 0  
    while i < 100:
        A = A * (np.dot(X,t(S))/ np.dot(np.dot(A,S),t(S)))
        B = B * (np.dot((L * Y), t(S)))/np.dot((L * np.dot(B,S)), t(S))
        S = S * (np.dot(t(A),X) + lamb*np.dot(t(B),L*Y))/(np.dot(t(A),np.dot(A,S)) + lamb*np.dot(t(B),L*np.dot(B,S)))
        i +=1
        if i%20 == 0:
            print i
    return A,S,B
def t(x):
    return np.transpose(x)

