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
    # print(tokenizer.__doc__)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        #i = 5
        #while i < len(token_list):# 
        for i in range(0, len(token_list)-50,5):#,random.randint(1,10)):
            n_gram_sequence = token_list[i:i+50]
            input_sequences.append(n_gram_sequence)
            #i = i + random.randint(3,7)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequence_len, padding='pre'))
    # print(input_sequences)
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len, total_words


def dataset_preparation_validation(data, max_len):
    corpus = data.lower().split("\n")
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(0, len(token_list)-50,5):
            n_gram_sequence = token_list[i:i+50]
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
    model.add(Embedding(total_words, 50, input_length=input_len))
    model.add(LSTM(units=25))
    model.add(Dropout(0.4))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def generate_text(seed_text, next_words, max_sequence_len, model):
    seed_text = seed_text[:50]
    output_seed = seed_text[:50]	
    print(output_seed)
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        #token_list = token_list/float(total_words)
        token_list = token_list.reshape((token_list.shape[0], token_list.shape[1], 1))
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = " "
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seeds = seed_text.split(" ")
        seeds[:-1] = seeds[1:]
        seeds[-1] = output_word
        seed_text = " "
        seed_text.join(seeds)
        output_seed += " " + output_word
    return output_seed


if FILE_DUMP:
    file = open('simple-examples/data/ptb.char.train.txt', 'r')
    data = file.read()
    X, Y, max_len, total_words = dataset_preparation(data)
    print(X.shape)
    val_file = open('simple-examples/data/ptb.char.valid.txt', 'r')
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

seed = "p o r t s _ o f _ c a l l _ i n c . _ r e a c h e d _ a g r e e m e n t s _ t o _ s e l l _ i t s _ r e m a i n i n g _ s e v e n _ a i r c r a f t _ t o _ b u y e r s _ t h a t _ w e r e _ n ' t _ d i s c l o s e d _ t h e _ a g r e e m e n t s _ b r i n g _ t o _ a _ t o t a l _ o f _ n i n e _ t h e _ n u m b e r _ o f _ p l a n e s _ t h e _ t r a v e l _ c o m p a n y _ h a s _ s o l d _ t h i s _ y e a r _ a s _ p a r t _ o f _ a _ r e s t r u c t u r i n g  _ t h e _ c o m p a n y _ s a i d _ a _ p o r t i o n _ o f _ t h e _ $ _ N _ m i l l i o n _ r e a l i z e d _ f r o m _ t h e _ s a l e s _ w i l l _ b e _ u s e d _ t o _ r e p a y _ i t s _ b a n k _ d e b t _ a n d _ o t h e r _ o b l i g a t i o n s _ r e s u l t i n g _ f r o m _ t h e _ c u r r e n t l y _ s u s p e n d e d _ < u n k > _ o p e r a t i o n s "
prediction = generate_text(seed, 100, max_len, model)
print(prediction)
