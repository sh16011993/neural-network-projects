from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
from keras import optimizers, initializers
import numpy as np
import matplotlib.pyplot as plt
import bz2
import pickle
import random
import sys

FILE_DUMP = True
tokenizer = Tokenizer()
data = """The cat and her kittens
They put on their mittens,
To eat a Christmas pie.
The poor little kittens
They lost their mittens,
And then they began to cry.
O mother dear, we sadly fear
We cannot go to-day,
For we have lost our mittens."
"If it be so, ye shall not go,
For ye are naughty kittens."""

def dataset_preparation(data):
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        #i = 5
        #while i < len(token_list):# 
        for i in range(1, len(token_list)):#,random.randint(1,10)):
            n_gram_sequence = token_list[:i]
            input_sequences.append(n_gram_sequence)
            #i = i + random.randint(3,7)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len, total_words


def dataset_preparation_validation(data, max_len):
    corpus = data.lower().split("\n")
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i]
            input_sequences.append(n_gram_sequence)
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_len, padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label



def create_model(X, max_sequence_len, total_words, opt):
    input_len = max_sequence_len - 1
    model = Sequential()
    x, y = X.shape
    model.add(Embedding(total_words, 100, input_length=input_len))
    model.add(LSTM(units=15))
    model.add(Dropout(0.6))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def generate_text(seed_text, next_words, max_sequence_len, model):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        #token_list = token_list/float(total_words)
        #token_list = token_list.reshape((token_list.shape[0], token_list.shape[1], 1))
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = " "
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


if FILE_DUMP:
    file = open('simple-examples/data/ptb.train.txt', 'r')
    data = file.read()
    X, Y, max_len, total_words = dataset_preparation(data)
    print(X.shape)
    val_file = open('simple-examples/data/ptb.valid.txt', 'r')
    data = val_file.read()
    X_val, Y_val = dataset_preparation_validation(data, max_len)
    print(X_val.shape)
else:
    X_val = pickle.load(open("X_val.pkl", "rb"))
    Y_val = pickle.load(open("Y_val.pkl", "rb"))
    Y = pickle.load(open("Y.pkl", "rb"))
    X = pickle.load(open("X.pkl", "rb"))

opt = optimizers.rmsprop(lr=0.001)
model = create_model(X, max_len, total_words, opt)
x, y = X.shape
X1 = X.reshape((x, y, 1))
#X1 = X1/float(total_words)
x, y = X_val.shape
X_val1 = X_val.reshape((x, y, 1))
#X_val1 =X_val1/float(total_words)
history = model.fit(X, Y, batch_size=512, epochs=10, validation_data=(X_val, Y_val), verbose=1, shuffle=True)
pickle.dump(model, open("model_word1.pkl", "wb"))

loss = history.history['loss']
val_loss = history.history['val_loss']
perplexity_train = np.exp(loss)
perplexity_val = np.exp(val_loss)
plt.plot(perplexity_train)
plt.plot(perplexity_val)
plt.title('model perplexity')
plt.ylabel('perplexity')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

seed = "these stocks"
prediction = generate_text(seed, 100, max_len, model)
print(prediction)
