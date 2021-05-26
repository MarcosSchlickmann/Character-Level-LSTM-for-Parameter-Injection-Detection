from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras.layers import Embedding, Dense, Dropout, LSTM
import numpy as np
import io
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# registro_para_prever = load_data("registro_para_prever.txt")

#Creating the dataset
x = x_normal  + x_anomalous
#assigning indices to each character in the query string
tokenizer = Tokenizer(char_level=True) #treating each character as a token
tokenizer.fit_on_texts(x) #training the tokenizer on the text
char_index = tokenizer.word_index


#to see the list of characters with their indices:
# print(char_index)

# x = x_normal[:50] + x_anomalous[:50]
to_predict = x
# to_predict = [ x_normal[random.randint(0, 20000)], x_anomalous[random.randint(0, 20000)] ]
# to_predict = ["gethttp://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=jam�n+ib�rico&precio=85", "gethttp://localhost:8080/teste.jsp?id=20&nombre='+drop+table"]

#creating the numerical sequences by mapping the indices to the characters
sequences = tokenizer.texts_to_sequences(to_predict)
char_index = tokenizer.word_index
# print(char_index)
maxlen = 1000   #length of the longest sequence=input_length
xx = pad_sequences(sequences, maxlen=maxlen)

model = models.load_model('model/lstm-model.h5')
model.load_weights('model/lstm-weights.h5')
model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])

predictions = model.predict(xx, verbose=1)
# print(predictions)

normalized_predictions = []
for prediction in predictions:
    if prediction[0] > 0.80:
        normalized_predictions.append(1)
        continue
    normalized_predictions.append(0)
# print(normalized_predictions)

no_yy = [0 for i in range(x_normal_len)]
an_yy = [1 for i in range(x_anomalous_len)]
yy = no_yy + an_yy

# for i in range(100):
#     print(normalized_predictions[i], yy[i])

# for i in zip(prediction, yy):
# 	print(i)


# y_pred_bool = np.argmax(normalized_predictions, axis=1)
report = classification_report(yy, normalized_predictions)
print(report)
# file = open('data/predictions.csv', 'w')
# with file:    
#     write = csv.writer(file, delimiter='\t')
#     write.writerows(zip(x, prediction))

# print(len(xx))
# print(xx)
# print(prediction)
# print("{:.4f}, {:.4f}".format(prediction[0][0], prediction[1][0]))
