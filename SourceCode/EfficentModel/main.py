#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


#from modules import GST


# In[3]:



pathAudio={}

with open('../data/split-0/train_balance3.tsv') as ftrain:
#with open('../data/split-0/autotagging_moodtheme-train.tsv') as ftrain:    
    contents= ftrain.read().splitlines()
    for con in contents[1:]:
        con=con.split("\t")[3].split(".")[0].split("/")
        #print(con)
        pathAudio[int(con[1])]=con[0]
with open('../data/split-0/autotagging_moodtheme-validation.tsv') as fvalidation:
    contents= fvalidation.read().splitlines()
    for con in contents[1:]:
        con=con.split("\t")[3].split(".")[0].split("/")
        pathAudio[int(con[1])]=con[0]
with open('../data/split-0/autotagging_moodtheme-test.tsv') as ftest:
    contents= ftest.read().splitlines()
    for con in contents[1:]:
        con=con.split("\t")[3].split(".")[0].split("/")
        pathAudio[int(con[1])]=con[0]


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

#import gzip
#import pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#print("A lib")
# In[5]:


torch.set_num_threads(7)


# ## Test small dataset

# In[6]:


import glob
listdata=glob.glob("/home/trinhan/AILAB/Emotion/mediaeval-2019-moodtheme-detection/submission1/data/melspecs/*.npy")

dsample=[int(i.replace("/home/trinhan/AILAB/Emotion/mediaeval-2019-moodtheme-detection/submission1/data/melspecs/","").replace(".npy","")) for i in listdata]
print(dsample)


# In[ ]:

print("GO here")



# # Load data

# ## Labels

# In[7]:


import commons
#train_info = commons.read_file('./data/split-0/autotagging_moodtheme-train.tsv')
train_info = commons.read_file('../data/split-0/train_balance3.tsv')
validation_info = commons.read_file('../data/split-0/autotagging_moodtheme-validation.tsv')
test_info = commons.read_file('../data/split-0/autotagging_moodtheme-test.tsv')

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


# In[9]:


train_df = parse(train_info)
valid_df = parse(validation_info)
test_df = parse(test_info)


# In[10]:


# In[11]:


train_df = train_df.loc[:, sorted(train_df.columns.tolist())]
valid_df = valid_df.loc[:, sorted(train_df.columns.tolist())]
test_df = test_df.loc[:, sorted(train_df.columns.tolist())]

test_df = test_df.sort_index()
# In[12]:


#train_df.sample(2)


# # Define datasets and augmentation

# In[13]:

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


# In[17]:


train_transform = get_transforms(
    train=True,
    size=6590,
    wrap_pad_prob=0.5,
    resize_scale=(0.8, 1.0),
    resize_ratio=(1.7, 2.3),
    resize_prob=0.33,
    spec_num_mask=3,
    spec_freq_masking=0.15,
    spec_time_masking=0.20,
    spec_prob=0
)
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
        #sample = np_load_gz('./data/melspecs/{}.npy.gz'.format(this_file))
        #sample = np.load('./data/melspecs/{}.npy'.format(this_file))
        #print(this_file)
        #print(pathAudio[this_file])
        
        #print(this_file)
        
        #name = str(this_file).replace("_","")
        name = int(this_file)
        
        
        sample = np.load('../../mtg-jamendo-dataset/moodtheme_mel/'+str(pathAudio[name])+'/'+str(name)+'.npy')
        image = sample
                
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.Tensor(self.df.values[idx])


# In[19]:


train_dataset = AudioDataset(train_df, train_transform)
valid_dataset = AudioDataset(valid_df, val_transform)
test_dataset = AudioDataset(test_df, val_transform)

print(len(train_dataset))

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses
    num = 0
    for item in images:   
        #print(item[1])
        for i in range(len(item[1])):
            if item[1][i] == 1:
                count[i] += 1
        #num += 1
        #if min(count) != 0: break

    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):  
        mx = 0
        for i in range(len(val[1])):
            if val[1][i] == 1:
                mx = max(mx, weight_per_class[i])
        weight[idx] = mx         
        #if idx > num: break


    return weight

#weights = make_weights_for_balanced_classes(train_dataset, 56) 
#print(weights)

#weights = torch.DoubleTensor(weights)
#sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


# In[20]:


#train_dataset[100][0].shape, train_dataset[100][1].shape


# In[21]:


#!conda install seaborn --y


# In[22]:


#import seaborn as sns
#sns.heatmap(train_dataset[100][0][0,:,:].numpy());


# In[23]:


#sns.heatmap(valid_dataset[1][0][0,:,:].numpy());


# # Model

# In[24]:


#import torchvision.models
from efficientnet_pytorch import EfficientNet
#import timm 
from modules import GST


# In[25]:


class Model(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.batch_normal = nn.BatchNorm2d(1)
        
        self.bw2col = nn.Sequential(
            nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())
        
        #self.mv2 = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)

        self.mv2 = EfficientNet.from_pretrained('efficientnet-b0', num_classes=56)
        #self.mv2_2 = EfficientNet.from_pretrained('efficientnet-b0', num_classes=56)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 56),
        )
        
        self.classifier_2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 56),
        )
        
        hparams = create_hparams()
        self.gst = GST(hparams)
    
    def forward(self, x):
        
        x = self.batch_normal(x)
        x = self.bw2col(x)
        #x1,x2 = x[:,:,:48,:],x[:,:,48:,:]
        #print(x.shape)
        
        '''cls = self.bw2col(x)
        cls = self.mv2.extract_features(cls)
        cls = cls.mean([2, 3])
        cls = self.classifier(cls)
        '''
        
        #x1,x2 = self.mv2_1(x1),self.mv2_2(x2)
        #return (x1 + x2)/2

        x = self.mv2(x)
        return x
        
        
## Training
# In[26]:

cuda=True
device = torch.device('cuda:0' if cuda else 'cpu')
print('Device: ', device)


# In[27]:


from hparams import create_hparams
model = Model().to(device)

#S = torch.load('gst/model')
#model.load_state_dict(S)

# In[28]:


optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)


# In[29]:

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        
        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        #targets = targets.type(torch.long)
        
        
        #print(targets)
        
        y = torch.ones(targets.shape).cuda()
        targets = torch.where(targets == 0, targets, y)
        
        targets = targets.type(torch.long)
        
        #print(targets)
        
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        
        
        #print(targets)
        #print(at)
        at = at.view(-1, 56)
        #print(at)
        
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss
        
#criterion = nn.BCEWithLogitsLoss(reduction='none')
#criterion = FocalLoss(gamma=2,logits=True,reduce=False)
criterion = WeightedFocalLoss(gamma=2)

# ----

# In[30]:


val_loader = DataLoader(valid_dataset, 40, shuffle=False, num_workers=4)
train_loader_1 = DataLoader(train_dataset, 8, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
train_loader_2 = DataLoader(train_dataset, 8, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)


# In[31]:


epochs = 120


# In[ ]:


train_loss_hist = []
train_loss_hist_each_class = []

valid_loss_hist = []
valid_loss_hist_each_class = []

lowest_val_loss = np.inf
lowest_val_loss_each_class = np.array([float('inf') for _ in range(56)])
epochs_without_new_lowest = 0

outdir = 'gst_only_eff_not_spec'
#model.load_state_dict(torch.load('./gst_only_eff_focal_gamma3/model'))
'''
for i in range(epochs):
    print('Epoch: ', i)
    
    this_epoch_train_loss = 0
    this_epoch_train_loss_each_class = np.zeros(56)
    for i1, i2 in zip(train_loader_1, train_loader_2):
        
        # mixup---------
        alpha = 1
        mixup_vals = np.random.beta(alpha, alpha, i1[0].shape[0])
        
        lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1, 1, 1))
        inputs = (lam * i1[0]) + ((1 - lam) * i2[0])
        
        lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1))
        labels = (lam * i1[1]) + ((1 - lam) * i2[1])
        # mixup ends ----------
        
        # https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-works-in-dataloader
        inputs = inputs.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)
        
        #print(labels)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            model = model.train()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss = loss.mean(dim=0).mean(dim=0)
            total_loss.backward()
            optimizer.step()
            
            loss_val = total_loss.detach().cpu().numpy()
            loss_val_each_class = loss.mean(dim=0).detach().cpu().numpy()
            
            #print("Batch: ", loss_val)
            
            this_epoch_train_loss += loss_val
            this_epoch_train_loss_each_class += loss_val_each_class
    
    this_epoch_valid_loss = 0
    this_epoch_valid_loss_each_class = np.zeros(56)
    for inputs, labels in val_loader:
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            model = model.eval()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            loss_val = loss.mean(dim=0).mean(dim=0).detach().cpu().numpy()
            loss_val_each_class = loss.mean(dim=0).detach().cpu().numpy()
            
            this_epoch_valid_loss += loss_val
            this_epoch_valid_loss_each_class += loss_val_each_class
    
    this_epoch_train_loss /= len(train_loader_1)
    this_epoch_train_loss_each_class /= len(train_loader_1)
    
    this_epoch_valid_loss /= len(val_loader)
    this_epoch_valid_loss_each_class /= len(val_loader)
    
    train_loss_hist.append(this_epoch_train_loss)
    train_loss_hist_each_class.append(this_epoch_train_loss_each_class)
    valid_loss_hist.append(this_epoch_valid_loss)
    valid_loss_hist_each_class.append(this_epoch_valid_loss_each_class)
    
    if this_epoch_valid_loss < lowest_val_loss:
        lowest_val_loss = this_epoch_valid_loss
        torch.save(model.state_dict(), './{}/model'.format(outdir))
        epochs_without_new_lowest = 0
    else:
        epochs_without_new_lowest += 1
    
    """for i in range(56):
        if this_epoch_valid_loss_each_class[i] < lowest_val_loss_each_class[i]:
            lowest_val_loss_each_class[i] = this_epoch_valid_loss_each_class[i]
            torch.save(model.state_dict(), './{}/model_c{}'.format(outdir,i))"""
    
    if epochs_without_new_lowest >= 25:
        break
    
    print(this_epoch_train_loss, this_epoch_valid_loss)
    
    scheduler.step(this_epoch_valid_loss)
'''

# # Final evaluation using the overall best model

# In[ ]:


model.load_state_dict(torch.load('./{}/model'.format(outdir)))

from sklearn.metrics import roc_auc_score, average_precision_score
# In[ ]:


#val_loader = DataLoader(valid_dataset, 32, shuffle=False, num_workers=6)
test_loader = DataLoader(test_dataset, 40, shuffle=False, num_workers=6)

# In[ ]:


this_epoch_valid_loss = 0
all_outputs = []
all_inputs = []

for inputs, labels in val_loader:
    
    #all_inputs.append(labels)
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        model = model.eval()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss_val = loss.mean(dim=0).mean(dim=0).detach().cpu().numpy()
        this_epoch_valid_loss += loss_val
        
        all_outputs.append(outputs.detach().cpu().numpy())
        all_inputs.append(labels.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
all_outputs = 1 / (1 + np.exp(-all_outputs))

all_inputs  = np.concatenate(all_inputs,  axis=0)

from sklearn.metrics import precision_recall_curve

thresholds = {}
for i in range(56):
    precision, recall, threshold = precision_recall_curve(all_inputs[:, i], all_outputs[:, i])
    f_score = np.nan_to_num((2 * precision * recall) / (precision + recall))
    thresholds[i] = threshold[np.argmax(f_score)]

all_outputs = []
all_inputs = []

for inputs, labels in test_loader:
    
    #all_inputs.append(labels)
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        model = model.eval()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss_val = loss.mean(dim=0).mean(dim=0).detach().cpu().numpy()
        this_epoch_valid_loss += loss_val
        
        all_outputs.append(outputs.detach().cpu().numpy())
        all_inputs.append(labels.detach().cpu().numpy())

all_inputs  = np.concatenate(all_inputs,  axis=0)

#pred1 = np.load("gst_only_eff_balanced_data_v3/predictions_ex.npy")
#pred2 = np.load("../WN/wavenet_feature_efficient_highest_b/predictions_ex.npy")
#all_outputs = 0.7 * pred1 + 0.3 * pred2

# In[ ]:


all_outputs  = np.concatenate(all_outputs,  axis=0)

#all_outputs.shape
#all_inputs.shape


# In[ ]:




# In[ ]:


from sklearn.metrics import roc_auc_score, average_precision_score


# In[ ]:


print(roc_auc_score(all_inputs, all_outputs, average='macro'))


# In[ ]:


print(average_precision_score(all_inputs, all_outputs, average='macro'))


# In[ ]:


#pd.DataFrame.from_records([
#    (i, roc_auc_score(all_inputs[:, i], all_outputs[:, i]).round(2))
#    for i in range(56)
#])

#np.save('{}/predictions_ex.npy'.format(outdir), all_outputs)


print(average_precision_score(all_inputs, all_outputs, average='macro'))

from sklearn.metrics import precision_recall_curve

decisions = []
for i in range(56):
    decisions.append(
        all_outputs[:, [i]] > thresholds[i]
    )
decisions = np.concatenate(decisions, axis=-1)

np.save('{}/predictions.npy'.format(outdir), all_outputs)
np.save('{}/decisions.npy'.format(outdir), decisions)
    

# # Final evaluation using the best model for each class

# In[ ]:

'''final_inputs = []
final_outputs = []
for i in range(56):
    model.load_state_dict(torch.load('./{}/model_c{}'.format(outdir, i)))
    val_loader = DataLoader(valid_dataset, 32, shuffle=False, num_workers=6)
    this_epoch_valid_loss = 0
    all_outputs = []
    all_inputs = []
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            model = model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_val = loss.mean(dim=0).mean(dim=0).detach().cpu().numpy()
            this_epoch_valid_loss += loss_val
            all_outputs.append(outputs.detach().cpu().numpy())
            all_inputs.append(labels.detach().cpu().numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_inputs  = np.concatenate(all_inputs,  axis=0)
    final_inputs.append(all_inputs[:, [i]])
    final_outputs.append(all_outputs[:, [i]])


# In[ ]:


all_inputs = np.concatenate(final_inputs, axis=-1)
all_outputs = np.concatenate(final_outputs, axis=-1)


# In[ ]:


all_outputs = 1 / (1 + np.exp(-all_outputs))


# In[ ]:


from sklearn.metrics import roc_auc_score, average_precision_score


# In[ ]:


print(roc_auc_score(all_inputs, all_outputs, average='macro'))


# In[ ]:


print(average_precision_score(all_inputs, all_outputs, average='macro'))


# In[ ]:


pd.DataFrame.from_records([
    (i, roc_auc_score(all_inputs[:, i], all_outputs[:, i]).round(2))
    for i in range(56)
])


# # Calculate optimal decision thresholds for each class

# In[ ]:


from sklearn.metrics import precision_recall_curve


# In[ ]:


# Optimized macro F-score
thresholds = {}
for i in range(56):
    precision, recall, threshold = precision_recall_curve(all_inputs[:, i], all_outputs[:, i])
    f_score = np.nan_to_num((2 * precision * recall) / (precision + recall))
    thresholds[i] = threshold[np.argmax(f_score)]


# # Create submission

# In[ ]:


final_outputs = []
for i in range(56):
    model.load_state_dict(torch.load('./{}/model_c{}'.format(outdir,i)))
    test_loader = DataLoader(test_dataset, 32, shuffle=False, num_workers=6)
    all_outputs = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            model = model.eval()
            outputs = model(inputs)
            all_outputs.append(outputs.detach().cpu().numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    final_outputs.append(all_outputs[:, [i]])


# In[ ]:


all_outputs = np.concatenate(final_outputs, axis=-1)


# In[ ]:


all_outputs = 1 / (1 + np.exp(-all_outputs))


print(roc_auc_score(all_inputs, all_outputs, average='macro'))
print(average_precision_score(all_inputs, all_outputs, average='macro'))



pd.read_csv('./moodtheme_split.txt', header=None).iloc[:, 0].str[13:].tolist() == sorted(train_df.columns.tolist())


# In[ ]:


all_outputs.shape, all_outputs.astype('float64').dtype


# In[ ]:


np.save('{}/predictions_b0_gst.npy'.format(outdir), all_outputs.astype('float64'))


# In[ ]:


decisions = []
for i in range(56):
    decisions.append(
        all_outputs[:, [i]] > thresholds[i]
    )
decisions = np.concatenate(decisions, axis=-1)


# In[ ]:


decisions.shape, decisions.dtype


# In[ ]:


np.save('{}/decisions_b0_gst.npy'.format(outdir), decisions)


# In[ ]:


#EfficientNet.from_pretrained('efficientnet-b1')


# In[ ]:





# In[ ]:





# In[ ]:
'''



