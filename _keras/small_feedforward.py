#!/usr/bin/env python
'''Trains and evaluate a simple MLP
on char ngram for classification task.
'''
from __future__ import print_function

import numpy as np
import keras

from time import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, GlobalAveragePooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard

max_words = 100000
batch_size = 32
epochs = 5
embedding_dims=64
maxlen=100

def load_data(file_path, tok, class_dict, do_fit=False):
    xs = []; ys = []
    with open(file_path) as f:
        for line in f:
            whitespace = line.find(" ")
            y = line[:whitespace]
            x = line[whitespace:]
            xs.append(x)
            ys.append(y)
    if do_fit:
        tok.fit_on_texts(xs)
        id = 1
        for y in ys:
            if y not in class_dict:
                class_dict[y] = id
                id += 1
            class_id = class_dict[y]
    xs = tok.texts_to_sequences(xs)
    ys = [class_dict[y] if y in class_dict else 0 for y in ys]
    return (xs, ys)

print('Loading data...')
class_dict = {}
tok = Tokenizer(num_words=max_words)
(x_train, y_train) = load_data("../dataset-3/snippets.train", tok, class_dict,True)  
(x_test, y_test) = load_data("../dataset-3/snippets.test", tok, class_dict)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
#print('Vectorizing sequence data...')
#x_train = tok.sequences_to_matrix(x_train, mode='binary')
#x_test = tok.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

#import pdb; pdb.set_trace()
print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_words,
                    embedding_dims,
                    input_length=maxlen))

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[tensorboard],
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])
