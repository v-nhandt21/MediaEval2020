import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import commons

pathAudio={}

#with open('../data/split-0/balance_multi.tsv') as ftrain:
with open('../data/split-0/train_balance3.tsv') as ftrain:    
    contents= ftrain.read().splitlines()
    for con in contents[1:]:
        con=con.split("\t")[3].split(".")[0].split("/")
        #print(con)
        pathAudio[int(con[1])]=con[0]
with open('../data/split-0/autotagging_moodtheme-validation_wn.tsv') as fvalidation:
    contents= fvalidation.read().splitlines()
    for con in contents[1:]:
        con=con.split("\t")[3].split(".")[0].split("/")
        pathAudio[int(con[1])]=con[0]
with open('../data/split-0/autotagging_moodtheme-test_wn.tsv') as ftest:
    contents= ftest.read().splitlines()
    for con in contents[1:]:
        con=con.split("\t")[3].split(".")[0].split("/")
        pathAudio[int(con[1])]=con[0]


# In[8]:


def parse(a_info):
    ans = []
    total = 0
    
    #print(a_info[])
    
    for k,v in a_info[1]['mood/theme'].items():
        
        #print(dsample)
        #vtm=[vv for vv in list(v) if vv in dsample]
        total += len(list(v))
        
        ans.append(pd.DataFrame({
            k: 1,
            #'file': list(vtm)
            'file': list(v)
        }))
    #print("total ", total)
        
    df = ans[0]
    
    #print(df)
    
    for x in ans[1:]:
        df = pd.merge(df, x, on='file', how='outer')
    df = df.set_index('file').fillna(0)
    df.loc[:, :] = df.loc[:, :].astype('int64')
    return df

train_info = commons.read_file('../data/split-0/train_balance3.tsv')
test_info = commons.read_file('../data/split-0/autotagging_moodtheme-test.tsv')

train_df = parse(train_info)
test_df = parse(test_info)

train_df = train_df.loc[:, sorted(train_df.columns.tolist())]
test_df = test_df.loc[:, sorted(train_df.columns.tolist())]

test_df = test_df.sort_index()

def np_load_gz(x):
    with gzip.open(x, 'rb') as f:
        return np.load(f)


# In[14]:


import random


# In[15]:


def random_30s(x):
    x_len = x.shape[1]
    offset = random.randint(0, (x_len - 1407))
    return x[:, offset:(offset + 1407)]


# In[16]:


from transforms import get_transforms
val_transform = get_transforms(False, 6590)


# In[18]:


class AudioDataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        
        this_file = self.df.index[idx]
        this_file = this_file.replace("i","").replace("m","")
        
        name = int(this_file)
        
        
        #sample = np.load('../../mtg-jamendo-dataset/moodtheme_wn/'+str(name)+'.npy')
        #image = sample
                
        #if self.transform is not None:
        #    image = self.transform(image)
        
        return torch.Tensor(np.array([0])), torch.Tensor(self.df.values[idx])



test_dataset = AudioDataset(test_df, val_transform)
test_loader = DataLoader(test_dataset, 40, shuffle=False, num_workers=6)

# In[ ]:


all_outputs = []
all_inputs = []
for inputs, labels in test_loader:
    
    all_inputs.append(labels)
    
    
all_inputs  = np.concatenate(all_inputs,  axis=0)

from sklearn.metrics import roc_auc_score, average_precision_score
    

all_outputs_1 = np.load("gst_only_eff_balanced_data_v3/predictions_ex.npy")
print("First prediction")

print(roc_auc_score(all_inputs, all_outputs_1, average='macro'))
print(average_precision_score(all_inputs, all_outputs_1, average='macro'))


all_outputs_2 = np.load("../WN/wavenet_feature_efficient_highest_b_bs32/predictions_ex.npy")
print("Second prediction")

print(roc_auc_score(all_inputs, all_outputs_2, average='macro'))
print(average_precision_score(all_inputs, all_outputs_2, average='macro'))


all_outputs = 0.7* all_outputs_1 + 0.3 * all_outputs_2
print("Ensemble prediction")
    
print(roc_auc_score(all_inputs, all_outputs, average='macro'))
print(average_precision_score(all_inputs, all_outputs, average='macro'))
