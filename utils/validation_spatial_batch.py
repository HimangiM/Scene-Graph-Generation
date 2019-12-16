
# coding: utf-8

# In[14]:


import json
from numpy.core import multiarray
import pickle as pkl
import torch
import math


# In[15]:


_dir = "/home/himangi/8th-Sem/major-project/Scene-Graph-Generation/"



# In[16]:


def create_batch_ids():
    with open(_dir + "vrd-dataset/test.pkl", 'rb') as infile:
        data_test = pkl.load(infile, encoding='latin1')
    
    x = []
    y = []
    sum_check = []
    none_check = 0
    
    for counter, img_data in zip(range(500), data_test):
        if img_data != None:
#             print (img_data)            
            sum_check.append(len(img_data['boxes']))
            for i, j, k in zip(img_data['ix1'], img_data['ix2'], img_data['rel_classes']):
                l_x = []
                l_x.append([bb for bb in img_data['boxes'][i]])
                l_x.append([bb for bb in img_data['boxes'][j]])
                x.append(l_x)
                y.append(k[0])
        else:
            none_check = none_check + 1
#         break
    
    print (sum(sum_check), len(x), len(y), none_check)
    return (x, y)


# In[17]:


def create_spatial_features_tensors(x_bb, y_bb):
    
    x = []
    y = []
    for sub_obj in x_bb:
        l = []
        sub = sub_obj[0]
        obj = sub_obj[1]
        
#         lx = (sub[0] - obj[0])/obj[0]
#         ly = (sub[1] - obj[1])/obj[1]
#         lw = math.log(sub[2]/obj[2])
#         lh = math.log(sub[3]/obj[3])
        
        lx = min(sub[0], obj[0])
        ly = min(sub[1], obj[1])
        lw = max(sub[0] + sub[2], obj[0] + obj[2]) - lx
        lh = max(sub[1] + sub[3], obj[1] + obj[2]) - ly
        
        l.append(lx)
        l.append(ly)
        l.append(lw)
        l.append(lh)
        
        x.append(l)
        
    x = torch.Tensor(x)
    
    for i in y_bb:
        l = [0 for j in range(0, 70)]
        l[i] = 1
        y.append(l)
        
    y = torch.Tensor(y)
    
#     print (x[0:5])
    print (x.shape, y.shape)
    
    torch.save(x, _dir + 'data/spatial_validation_x.pt')
    torch.save(y, _dir + 'data/spatial_validation_y.pt')


# In[18]:


if __name__=='__main__':
    x_bb, y_bb = create_batch_ids()
    create_spatial_features_tensors(x_bb, y_bb)

