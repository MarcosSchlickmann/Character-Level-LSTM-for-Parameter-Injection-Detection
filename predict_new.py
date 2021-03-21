from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras.layers import Embedding, Dense, Dropout, LSTM
import numpy as np
import io
from sklearn.model_selection import train_test_split

import random


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


x_normal = load_data("normal_parsed.txt")
x_anomalous = load_data("anomalous_parsed.txt")

# registro_para_prever = load_data("registro_para_prever.txt")

#Creating the dataset
x = x_normal  + x_anomalous

#assigning indices to each character in the query string
tokenizer = Tokenizer(char_level=True) #treating each character as a token
tokenizer.fit_on_texts(x) #training the tokenizer on the text

to_predict = [ x_normal[random.randint(0, 20000)], x_anomalous[random.randint(0, 20000)] ]
print(to_predict)

#creating the numerical sequences by mapping the indices to the characters
sequences = tokenizer.texts_to_sequences(to_predict)
char_index = tokenizer.word_index
print(char_index)
maxlen = 1000   #length of the longest sequence=input_length
xx = pad_sequences(sequences, maxlen=maxlen)

model = models.load_model('securitai-lstm-model.h5')
model.load_weights('securitai-lstm-weights.h5')
model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])

prediction = model(xx)
print(len(xx))
print(xx)
print(prediction)
