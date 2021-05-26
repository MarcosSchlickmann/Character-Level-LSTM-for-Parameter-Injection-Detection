import sys
from keras import preprocessing
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras.layers import Embedding, Dense, Dropout, LSTM
import numpy as np
import io
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import json

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


x_normal = load_data("data/normal_parsed.txt")
x_anomalous = load_data("data/anomalous_parsed.txt")
x_normal_len = len(x_normal)
x_anomalous_len = len(x_anomalous)
x = x_normal  + x_anomalous

no_yy = [0 for i in range(x_normal_len)]
an_yy = [1 for i in range(x_anomalous_len)]
y = no_yy + an_yy

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
print('len x_train: {}, len y_train: {}'.format(len(x_test), len(y_test)))

with open('data/tokenized-chars.json') as json_file:
    tokenizer_conf = json.load(json_file)
tokenizer = tokenizer_from_json(tokenizer_conf)
char_index = tokenizer.word_index

to_predict = x_test

#creating the numerical sequences by mapping the indices to the characters
sequences = tokenizer.texts_to_sequences(to_predict)
char_index = tokenizer.word_index
maxlen = 1000   #length of the longest sequence=input_length
xx = pad_sequences(sequences, maxlen=maxlen)

model = models.load_model('model/lstm-model.h5')
model.load_weights('model/lstm-weights.h5')
model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])

predictions = model.predict(xx, verbose=1)

normalized_predictions = []
default_predictions = []
for prediction in predictions:
    default_predictions.append(round(prediction[0], 4))
    if prediction[0] > 0.80:
        normalized_predictions.append(1)
        continue
    normalized_predictions.append(0)


report = classification_report(y_test, normalized_predictions)
print(report)
tn, fp, fn, tp = confusion_matrix(y_test, normalized_predictions).ravel()
print("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp))

def format_row(a,b,c):
    return ('{:.4f}'.format(a), b, c)

file = open('data/predictions2.csv', 'w')
with file:    
    write = csv.writer(file, delimiter=',')
    write.writerows([format_row(a,b,c) for (a,b,c) in zip(default_predictions, normalized_predictions, y_test)])
