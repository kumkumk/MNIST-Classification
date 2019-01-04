#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
import math
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
import gzip


# In[2]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# In[3]:


#print(training_data[1].shape)#10l test data
tr_data=training_data[0]
tr_target=training_data[1]
val_data=validation_data[0]
val_target=validation_data[1]
tst_data=test_data[0]
tst_target=test_data[1]


# In[4]:


#one hot encoding

def oneHotEncode(t): 
    l1=[]
    for i in range(len(t)):
        l=[0,0,0,0,0,0,0,0,0,0]
        l[t[i]]=1
        l1.append(l)
    t=l1    
    return t


# In[5]:


tr_target=oneHotEncode(tr_target)
val_target=oneHotEncode(val_target)
tst_target=oneHotEncode(tst_target)


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


# In[20]:


print(USPSTar[10000:10050])


# ## Processing Input and Label Data

# In[8]:



processedTrainingData   = tr_data
processedTrainingLabel = tr_target


# In[9]:


processedTestingData    = tst_data
processedTestingLabel=tst_target


# In[10]:


processedTestingDataUSPS=USPSMat
processedTestingLabelUSPS=USPSTar


# ## Tensorflow Model Definition

# In[11]:


# Defining Placeholder
inputTensor  = tf.placeholder(tf.float32, [None, 784])#shape 784 for 28X28 flattened image
outputTensor = tf.placeholder(tf.float32, [None, 10])#shape 10 for 10 class labels


# In[12]:


NUM_HIDDEN_NEURONS_LAYER_1 = 100
#NUM_HIDDEN_NEURONS_LAYER_2 = 80
LEARNING_RATE = 0.08
#keep_prob = 0.08

# Initializing the weights to Normal Distribution
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

# Initializing the input to hidden layer weights
input_hidden_weights  = init_weights([784, NUM_HIDDEN_NEURONS_LAYER_1])

# Initializing the second hidden layer weights
#intermed_hidden_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1,NUM_HIDDEN_NEURONS_LAYER_2])

# Initializing the hidden to output layer weights
hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 10])

# Computing values at the hidden layer
hidden_layer = tf.nn.sigmoid(tf.matmul(inputTensor, input_hidden_weights))

#adding dropout to the hidden layer
#hidden_layers = tf.nn.dropout(hidden_layer, keep_prob)

# Computing values at the second hidden layer
#hidden_layers = tf.nn.sigmoid(tf.matmul(hidden_layer, intermed_hidden_weights )) 

# Computing values at the output layer
output_layer = tf.matmul(hidden_layer, hidden_output_weights)

#use below output_layer1 for prediction in case of dropout
#output_layer1 = tf.matmul(hidden_layer, hidden_output_weights)

# Defining Error Function
error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))
#error_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

# Defining Learning Algorithm and Training Parameters - AdamOptimizer and Gradient Descent

#training = tf.train.AdamOptimizer(LEARNING_RATE,epsilon = 0.1).minimize(error_function)
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

# Prediction Function
prediction = tf.argmax(output_layer, 1)


# # Training the Model

# In[13]:


NUM_OF_EPOCHS = 100
BATCH_SIZE = 150

training_accuracy = []
error_list = []

with tf.Session() as sess:
    
    # Set Global Variables ?
    # all the variables declared must be initialised explicitly before they can be used in training
    tf.global_variables_initializer().run()
    
    for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):
        
        #Shuffle the Training Dataset at each epoch
        #p = np.random.permutation(range(len(processedTrainingData)))
        #processedTrainingData  = processedTrainingData[p]
        #processedTrainingLabel = processedTrainingLabel[p]
        
        # Start batch training
        for start in range(0, len(processedTrainingData), BATCH_SIZE):
            end = start + BATCH_SIZE
            #print(processedTrainingData.shape)
            #print(processedTrainingLabel.shape)
            result=sess.run([training, error_function], feed_dict={inputTensor: processedTrainingData[start:end], 
                                          outputTensor: processedTrainingLabel[start:end]})
            error_list.append(result[1])
            # result[0] => training
            # result[1] => error_function
        # Training accuracy for an epoch
        training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                             outputTensor: processedTrainingLabel})))
    # Testing
    predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData})
    predictedTestLabelUSPS = sess.run(prediction, feed_dict={inputTensor: processedTestingData})


# In[14]:


print(predictedTestLabel[:2])


# In[47]:


df = pd.DataFrame()
df['acc'] = training_accuracy
print('TrainingAccuracy '+str(training_accuracy[-1]*100))
df.plot(grid=True)
#plotting the Error function for each batch across epochs
df1 = pd.DataFrame()
df1['error'] = error_list
df1.plot(grid=True)


# In[48]:


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return 0
    elif encodedLabel == 1:
        return 1
    elif encodedLabel == 2:
        return 2
    elif encodedLabel == 3:
        return 3
    elif encodedLabel == 4:
        return 4
    elif encodedLabel == 5:
        return 5
    elif encodedLabel == 6:
        return 6
    elif encodedLabel == 7:
        return 7
    elif encodedLabel == 8:
        return 8
    elif encodedLabel == 9:
        return 9


# # Testing the Model [Software 2.0]

# In[49]:


wrong   = 0
right   = 0

predictedTestLabelList = []
""
for i,j in zip(processedTestingLabel,predictedTestLabel):
    predictedTestLabelList.append(decodeLabel(j))
    
    if np.argmax(i) == j:
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy on MNIST: " + str(right/(right+wrong)*100))


# In[50]:


wrong   = 0
right   = 0

predictedTestLabelList = []
""
for i,j in zip(processedTestingLabelUSPS,predictedTestLabelUSPS):
    predictedTestLabelList.append(decodeLabel(j))
    
    if np.argmax(i) == j:
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy on USPS: " + str(right/(right+wrong)*100))


# In[ ]:





# In[ ]:




