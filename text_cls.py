
#%%
# Import modules
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import TensorBoard
import os,datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import re
# %%
# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')
# %%
# Data Inspection
df.isna().sum()

# %%
df.info()
# %%
df.describe().sum()
# %%
df.duplicated().sum()
# %%
df = df.drop_duplicates()
# %%
X = df['text']
y = df['category']

# %%
temp = []
for index,txt in X.items():
    X[index] = re.sub('<.*>',' ',txt)
    X[index] = re.sub('[^a-zA-Z]',' ',X[index])
    temp.append(len(X[index].split()))
    #review[index] = re.sub('[^a-zA-Z]',' ',review[index]).lower()
# %%
# Data preprocessing (X)
num_words = 5000
oov_token = '<oov>'
token = Tokenizer(num_words = num_words , oov_token = oov_token)
token.fit_on_texts(X)
word_index = token.word_index
train_sequences = token.texts_to_sequences(X)

# %%
# Padding

train_sequences = pad_sequences(train_sequences,maxlen=100,padding='post',truncating='post')


# %%
# Target label(y) preprocessing
ohe = OneHotEncoder(sparse=False)
train_category = ohe.fit_transform(y[::,None])

#%%
train_sequences = np.expand_dims(train_sequences,-1)
#%%
#Train-test split
X_train,X_test,y_train,y_test = train_test_split(train_sequences,train_category,shuffle=True,random_state=12345)
# %%
#Model Development
model = Sequential()
#model.add(Input(X_train.shape[1:]))
model.add(Embedding(num_words,64))
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.summary()
# %%

plot_model(model,show_shapes=True) 
#%%
# Model compilation
model.compile(optimizer='adam',loss ='categorical_crossentropy', metrics = 'acc')
#callbacks
LOGS_PATH = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime('%Y&m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH)
hist = model.fit(X_train,y_train, callbacks = tensorboard_callback ,validation_data=(X_test,y_test), epochs=10)
# %%
#Model Evaluation
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)

print(classification_report(y_true,y_pred))
#%%
#Model Saving

# Save tokenizer

import json

with open('saved_models.json','w') as f:
    json.dump(token.to_json(),f)


# %% 
# save ohe

import pickle
with open('ohe.pkl','wb') as f:
    pickle.dump(ohe,f)

# save deep learning model

model.save('saved_models.h5')
# %%
