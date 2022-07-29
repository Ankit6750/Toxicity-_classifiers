#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install tensorflow tensorflow-gpu pandas matplotlib sklearn


# In[2]:


import os 
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


os.path.join("F:\Toxicity_classification","train",'train.csv')


# In[4]:


df=pd.read_csv(
    os.path.join("F:\Toxicity_classification","train","train.csv"))


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


from tensorflow.keras.layers import TextVectorization


# In[8]:


#TextVectorization??


# In[9]:


x=df['comment_text'] # separate comments
y=df[df.columns[2:]].values # seprate labels


# In[10]:


# generate text vextorization layer
max_features = 200000 # no of words in vocab
vectorize = TextVectorization(max_tokens=max_features,
                             output_sequence_length=1800,
                             output_mode='int')


# In[11]:


vectorize.adapt(x.values)


# In[12]:


vectorize('Hello world,life is great')[:5]


# In[13]:


vectorize_text=vectorize(x.values)


# In[14]:


#vectorize.get_vocabulary(include_special_tokens=False)


# In[15]:


vectorize_text # 159571 sentance and 1800 words shape of array


# In[16]:


# MCSBAP = map,chache,shuffle,batch,prefetch, from_tensorflow_slices,lest_file
# pipline steps 
dataset = tf.data.Dataset.from_tensor_slices((vectorize_text,y))
dataset = dataset.cache()
dataset = dataset.shuffle(16000) 
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # help for bottleneck


# In[17]:


batch_x,batch_y=dataset.as_numpy_iterator().next()


# In[18]:


batch_x,batch_y


# In[19]:


# split data into train test validation
train = dataset.take(int(len(dataset)*0.7))
val = dataset.skip(int(len(dataset)*0.7)).take(int(len(dataset)*.2)) # skip 70% and take 20% data
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1)) # skip 90% as train and vali data and take 10% as test data
train_data_generator = train.as_numpy_iterator()


# In[20]:


train_data_generator.next()


# ## Train model

# In[21]:


#Build deep learning model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Bidirectional,Dense,Embedding


# In[22]:


#Bidirectional??
#https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjA3NGI5MjhlZGY2NWE2ZjQ3MGM3MWIwYTI0N2JkMGY3YTRjOWNjYmMiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2NTg4OTQ1MDEsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwOTQxMTM0Njc2NTgwNjA0Nzg5MyIsImVtYWlsIjoiYWRhZGFyd2FsYTExQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJuYW1lIjoiQW5raXQgRGFkYXJ3YWxhIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BRmRadWNyc3liWFFCRDBNd0pESWlrUEx4X1hNVVhlREVDYWRLZHRMa1BLQXZ3PXM5Ni1jIiwiZ2l2ZW5fbmFtZSI6IkFua2l0IiwiZmFtaWx5X25hbWUiOiJEYWRhcndhbGEiLCJpYXQiOjE2NTg4OTQ4MDEsImV4cCI6MTY1ODg5ODQwMSwianRpIjoiZWIxMjcyNThmZGUyN2VkYmZiODNhOTNjZGI5YmRhMThlYTRiOWNiNiJ9.Pf173nW40MJsGxR5cJi6--j-k-cY_X-CqZE3vzKixcwMNW4dZpGw99oTQyspZA02Puqv-Iz5R92NLN3qZbqaGQtMEg0uPkHn9i-KR1_2YudxsVxU95kj2xi5h3VhS4R2AaAw5mAXH3kmNFDEWU8gLXuWFVbsWo86gpQkxUyU3AkCHkRNsNDqiEdQcSDIrHCBakeLwq4hTQseHjPzZZHc0b3LDeIC8PB7h_QEC9orfB9ZIc7oMMzyj3aLSrVaJzjxrWZsgahPbbO3OxtSMHp5wKkNUBKuw6U1JEdGUJTlKoNhQsIsLPDRoDWzlZBF2DsHXKqUdzXfCytJTZbmMjYysQ


# In[23]:


model = Sequential()
model.add(Embedding(max_features+1,32)) #Embedding layer enables us to convert each word into a fixed length vector of defined size
model.add(Bidirectional(LSTM(32,activation='tanh'))) # Bidirectional LSTM layer
# feature extraction layer
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
# Final layer
model.add(Dense(6,activation='sigmoid'))


# In[24]:


model.compile(loss='BinaryCrossentropy',optimizer='Adam')
model.summary()


# In[25]:


history = model.fit(train,epochs=3,validation_data=val,)


# In[26]:


import matplotlib.pyplot as plt
history.history 


# In[28]:


plt.figure(figsize=(8,6))
pd.DataFrame(history.history).plot()
plt.show()


# ## Make Prediction

# In[37]:


text = vectorize('Good person')
model.predict(np.expand_dims(text,0))


# In[38]:


df.columns[2:]


# In[39]:


batch = test.as_numpy_iterator().next()


# In[47]:


batch_x,batch_y = test.as_numpy_iterator().next()


# In[48]:


batch_y


# In[49]:


(model.predict(batch_x)>0.5).astype(int)


# ## Evaluate Model

# In[51]:


from tensorflow.keras.metrics import Precision,Recall,CategoricalAccuracy
pre=Precision()
re= Recall()
acc=CategoricalAccuracy()


# In[52]:


for batch in test.as_numpy_iterator():
    #Unpack the batch
    X_true,y_true = batch
    # Make prediction
    yhat = model.predict(X_true)
    
    # Flatten the prediction
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true,yhat)
    re.update_state(y_true,yhat)
    acc.update_state(y_true,yhat)


# In[53]:


print(f'Precision:{pre.result().numpy()},Recall:{re.result().numpy()},Accuracy:{acc.result().numpy()}')


# In[54]:


#!pip install gradio jinja2
import gradio as gr


# In[55]:


model.save(r'F:\Toxicity_classification\toxicity.h5')


# In[73]:


model = tf.keras.models.load_model(r'F:\Toxicity_classification\toxicity.h5')
input_str  = vectorize('I hate you fucking idiot')
res = model.predict(np.expand_dims(input_str,0))
(res >= 0.5).astype(int)


# In[74]:


model


# In[81]:


def score_comment(comment):
    vectorized_comment = vectorize([comment])
    result = model.predict(vectorized_comment)
    
    text= ''
    for idx,col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col,result[0][idx]>0.5)
      
    return text


# In[84]:


interface = gr.Interface(fn=score_comment,
                        inputs=gr.inputs.Textbox(lines=2,placeholder='comment to score'),
                        outputs='text')


# In[85]:


interface.launch(share=True)


# In[ ]:




