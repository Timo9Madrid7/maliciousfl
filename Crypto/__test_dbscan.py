#!/usr/bin/env python
# coding: utf-8

# # DBSCAN

# ## Plot Method

# In[1]:


from jupyterthemes import jtplot
jtplot.style("grade3")
from Crypto.dbscan import plot_2Dresult


# ## Generate sample data

# In[2]:


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)


# ## Implemented DBSCAN

# In[3]:


from Crypto.dbscan import DBSCAN as myDBSCAN
import torch

myCluster = myDBSCAN(0.3, 10).fit(torch.from_numpy(X))
core_sample_indices = myCluster.core_sample_indices_
mylabels = myCluster.labels_
plot_2Dresult(X, mylabels, core_sample_indices)


# ## SKLearn DBSCAN

# In[4]:


from sklearn.cluster import DBSCAN
from sklearn import metrics

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_sample_indices = db.core_sample_indices_
labels = db.labels_
plot_2Dresult(X, labels, core_sample_indices)


# In[ ]:




