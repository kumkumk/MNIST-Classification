#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import gzip
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[3]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# In[4]:


#print(training_data[1].shape)#10l test data
tr_data=training_data[0]
tr_target=training_data[1]
val_data=validation_data[0]
val_target=validation_data[1]
tst_data=test_data[0]
tst_target=test_data[1]


# In[5]:


tr_data=np.array(tr_data)
tr_target=np.array(tr_target)
print(tr_data.shape)
print(tr_target.shape)


# In[6]:


x_train = tr_data
y_train=tr_target
x_test=tst_data
y_test=tst_target


# In[7]:


from PIL import Image
import os
import numpy as np


# In[8]:


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


# In[9]:


#USPS data to be used for Testing
USPSMat=np.array(USPSMat)
USPSTar=np.array(USPSTar)


# In[10]:


clf=SVC(kernel='rbf',gamma='scale')
#clf=SVC(kernel='linear',C=0.5)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
#y_pred_USPS=clf.predict()
print('accuracy',metrics.accuracy_score(y_test,y_pred))


# In[11]:


conf_mat = confusion_matrix(y_test, y_pred)


# In[12]:


print(conf_mat)

