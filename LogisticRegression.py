#!/usr/bin/env python
# coding: utf-8

# ## Load MNIST on Python 3.x

# In[106]:


import pickle
import gzip
import numpy as np


# In[107]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# In[108]:


#print(training_data[1].shape)#10l test data
tr_data=training_data[0]
tr_target=training_data[1]
val_data=validation_data[0]
val_target=validation_data[1]
tst_data=test_data[0]
tst_target=test_data[1]


# In[109]:


tr_data.shape


# In[110]:


#one hot encoding

def oneHotEncode(t): 
    l1=[]
    for i in range(len(t)):
        l=[0,0,0,0,0,0,0,0,0,0]
        l[t[i]]=1
        l1.append(l)
    t=l1    
    return t
    


# In[111]:


tr_targetE=oneHotEncode(tr_target)
val_targetE=oneHotEncode(val_target)
tst_targetE=oneHotEncode(tst_target)


# In[112]:


len(tr_target)


# ## Load USPS on Python 3.x

# In[6]:


from PIL import Image
import os
import numpy as np


# In[7]:


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


# In[8]:


USPS_Tar=oneHotEncode(USPSTar)


# In[113]:


#Accuracy Calculation Function
def AccCalc(W,data,target):
    count=0
    final_pred=[]
    for i in range(len(data)):
        pred=np.dot(W,data[i])
        sum1=0
        for j in range(10):
            sum1=sum1+np.exp(pred[j])
        pred=np.exp(pred)
        pred=pred/sum1
        final_pred.append(np.argmax(pred))
        #print('predicted vector')
        #print(pred)
        #print('actual target')
        #print(tr_target[i])
        if(np.argmax(pred)==np.argmax(target[i])):
            count=count+1
    count1=(count/len(target))*100       
    return count1,final_pred


# In[114]:


W_init1=np.ones((10,784))
W_init=np.ones((10,784))


# In[116]:


c1=0
c2=0
c3=0
lamda=0.01
lr=0.01
for i in range(45000):
    #print('iteration'+str(i))
    h=np.dot(W_init,tr_data[i])
    #print(h)
    #print(W_init[:,400:550])
    #print(h)
    sm=0
    for j in range(10):
        sm=sm+np.exp(h[j])
    h=np.exp(h)   
    h=h/sm #softmax calculation0
    #print(h)    
    t=(h-tr_targetE[i]).reshape((10,1))
    d=tr_data[i].reshape((784,1))
    DeltaW=np.dot(t,np.transpose(d))
    
    W=W_init-lr*(DeltaW+(lamda*W_init))
    
    W_init=W
c1,pred1=AccCalc(W,tr_data,tr_targetE)
c2,pred2=AccCalc(W,val_data,val_targetE)
c3,pred3=AccCalc(W,tst_data,tst_targetE)
c4,pred4=AccCalc(W,USPSMat,USPS_Tar)
print('Train Accuracy')
print(c1)
print('Validation Accuracy')
print(c2)
print('Test Accuracy')
print(c3)
print('USPS Accuracy')
print(c4)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[121]:


conf_mat = confusion_matrix(tst_target, pred3)
print('Confusion Matrix')
conf_mat

