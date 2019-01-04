#!/usr/bin/env python
# coding: utf-8

# ## Load MNIST on Python 3.x

# In[12]:


import pickle
import gzip
import numpy as np


# In[13]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# In[14]:


#print(training_data[1].shape)#10l test data
tr_data=training_data[0]
tr_target=training_data[1]
val_data=validation_data[0]
val_target=validation_data[1]
tst_data=test_data[0]
tst_target=test_data[1]


# In[15]:


tr_data=np.array(tr_data)
tr_target=np.array(tr_target)
val_data=np.array(val_data)
val_target=np.array(val_target)
tst_data=np.array(tst_data)
tst_target=np.array(tst_target)


# In[16]:


tst_target=tst_target.reshape((10000,1))
val_target=val_target.reshape((10000,1))


# In[17]:


test_data=np.vstack((val_data,tst_data))
test_target=np.vstack((val_target,tst_target))


# ## Load USPS on Python 3.x

# In[20]:


from PIL import Image
import os
import numpy as np


# In[21]:


USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


# In[22]:


#USPS data to be used for Testing
USPSMat=np.array(USPSMat)
USPSTar=np.array(USPSTar)


# # Random Forest Classification

# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[24]:


clf = RandomForestClassifier(n_estimators=100, max_depth=20,random_state=0,min_samples_split=2)
Model=clf.fit(tr_data,tr_target)


# In[26]:


pred_tr=Model.predict(tr_data)# dont predict on train data
#pred_val=Model.predict(val_data)#merge these 2 into one
#pred_tst=Model.predict(tst_data)
pred = Model.predict(test_data)
pred_USPS=Model.predict(USPSMat)


# In[28]:


#Randomly validating predictions
for i in range(0, 5):
        print ("Actual outcome :: {} and Predicted outcome :: {}".format(tr_target[i], pred[i]))


# In[29]:


print ("Train Accuracy :: ", accuracy_score(tr_target, pred_tr))
print ("Test Accuracy :: on MNIST", accuracy_score(test_target, pred))
print ("Test Accuracy :: on USPS", accuracy_score(USPSTar, pred_USPS))


# In[30]:


conf_mat = confusion_matrix(test_target, pred)


# In[31]:


conf_mat


# In[ ]:




