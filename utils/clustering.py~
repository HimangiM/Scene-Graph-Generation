#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle as pkl
import math
import random
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler  


# In[ ]:


_dir = "/home/himangi/8th-Sem/major-project/Scene-Graph-Generation/"
with open(_dir + "data/param_emb_new_rep_dict.pkl", "rb") as infile:
    # param = pkl._Unpickler(infile)
    # param.encoding = 'latin1'
    param_data = pkl.load(infile)


# In[ ]:


labels = []
with open(_dir + "vrd-dataset/obj.txt", "r") as infile:
    for lines in infile.readlines():
        labels.append(lines.strip())


# In[ ]:


X_train = param_data[0:80]
X_test = param_data[80:]
y_train = labels[0:80]
y_test = labels[80:]


# In[ ]:


scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  


# In[ ]:


# model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
# model.fit(X_train,y_train)


# In[ ]:


# y_pred = model.predict(X_test)
# print (y_pred)


# In[ ]:


model = DBSCAN(eps=0.3)
model.fit(param_data)
X_tsne = TSNE().fit_transform(param_data)
fig=plt.figure()
# img=fig.add_subplot(111)
# plot=img.scatter(X_tsne[:,0],X_tsne[:,1])


# In[ ]:


#print (model.kneighbors(X=[param_data[0]]))
print ('Cluster image saved')
print ('Clusters saved')

# In[ ]:


#plt.savefig('./clusters/clustering.png')


# In[ ]:





