#!/usr/bin/python
# _*_ coding: utf-8 _*_

from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras.layers import Embedding, Dense, Dropout, LSTM
import numpy as np
import io
from sklearn.model_selection import train_test_split

#loading the parsed files
def load_data(file):
    with io.open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result


x_normal = load_data("data/normal_parsed.txt")
x_anomalous = load_data("data/anomalous_parsed.txt")

#Creating the dataset
x = x_normal  + x_anomalous

#creating labels normal=0, anomalous=1
y_normal = [0] * len(x_normal)
y_anomalous = [1] * len(x_anomalous)
y = y_normal + y_anomalous

#assigning indices to each character in the query string
tokenizer = Tokenizer(char_level=True) #treating each character as a token
tokenizer.fit_on_texts(x) #training the tokenizer on the text


#spliting the dataset into train and test 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

char_index = tokenizer.word_index
#to see the list of characters with their indices:
print(char_index)

#creating the numerical sequences by mapping the indices to the characters
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

maxlen = 1000   #length of the longest sequence=input_length

train_data = pad_sequences(train_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)


y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

x_train = train_data
x_test = test_data

#size of the vector space in which characters will be embedded
embedding_dim = 32

#size of the vocabulary or input_dim
max_chars = 63


def build_model():
    model = models.Sequential()
    model.add(Embedding(max_chars, embedding_dim, input_length=maxlen))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
print(model.summary())


#training the model on the entire training set and evaluating it using the testing data
model.fit(x_train, y_train, epochs=3, batch_size=32)
test_acc, test_loss = model.evaluate(x_test, y_test)
print(test_acc, test_loss)

model.save_weights('model/lstm-weights.h5')
model.save('model/lstm-model.h5')
