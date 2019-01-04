#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
#from __future__ import print_function


# In[2]:


from PIL import Image
import os
import numpy as np


# In[3]:


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


# In[4]:


Mat=np.array(USPSMat)
Tar=np.array(USPSTar)

Mat=Mat.reshape((-1,28,28,1))

Mat=Mat.astype('float32')
Mat /= 255


# In[5]:


Tar=np.array(Tar)
print(Tar.shape)


# In[6]:


batch_size = 128
num_classes = 10
epochs = 2


# In[7]:


# input image dimensions
img_x, img_y = 28, 28


# In[8]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[9]:


print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)


# In[10]:


x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)


# In[11]:


# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[12]:


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


# In[13]:


history = AccuracyHistory()


# In[14]:


#convert the target labels to 10 dimensional vector
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
Tar = keras.utils.to_categorical(Tar, num_classes)


# In[15]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),# using a 32 channel output(32 filters) for each data point
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))# using 64 channel output
model.add(MaxPooling2D(pool_size=(2, 2)))# max pooling
model.add(Flatten())# flatten the 2D output so as to feed into the output layer
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])# using cross entropy error and accuracy as metrics


# In[16]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
    callbacks=[history])


# In[17]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss on MNIST:', score[0])
print('Test accuracy percentage on MNIST:', score[1]*100)


# In[18]:


pred=model.predict(x_test)
matrix1=confusion_matrix(y_test.argmax(axis=1),pred.argmax(axis=1))
print(matrix1)


# In[19]:


score = model.evaluate(Mat, Tar, verbose=0)
print('Test loss on USPS:', score[0])
print('Test accuracy percentage on USPS:', score[1]*100)

