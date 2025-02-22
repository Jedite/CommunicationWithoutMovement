# -*- coding: utf-8 -*-
"""SuggestionNext.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12D0IBF6a9Fgf51kyoetJfS6KE1SKKpvm
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils.np_utils as ku
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.models import load_model, Sequential
from keras.layers import Bidirectional, LSTM, Embedding, Dense, Dropout, Flatten, Input, CuDNNLSTM, Conv1D, MaxPooling1D, LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2
from keras import Model
import keras

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize
from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import numpy as np
import string
import re
import tarfile
import gc
import os
import sys

gc.enable()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

"""###Load, filter, and encode text and labels###"""

MAX_SAMPLES = 10000
EMBED_DIM = 300
LATENT_DIM = 512
NUM_HEADS = 8
MAX_LENGTH = 40

path_to_zip = tf.keras.utils.get_file(
    "cornell_movie_dialogs.zip",
    origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
    extract=True,
)

path_to_dataset = os.path.join(
    os.path.dirname(path_to_zip), "cornell movie-dialogs corpus"
)
path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
path_to_movie_conversations = os.path.join(path_to_dataset, "movie_conversations.txt")


def load_conversations():
    # Helper function for loading the conversation splits
    id2line = {}
    with open(path_to_movie_lines, errors="ignore") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(path_to_movie_conversations, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
        for i in range(len(conversation) - 1):
            inputs.append(id2line[conversation[i]])
            outputs.append(id2line[conversation[i + 1]])
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs


questions, answers = load_conversations()

text = ' '.join(answers)
print(text)

ex = open('./1661-0.txt').read().lower()
print(ex)

extokenizer = RegexpTokenizer(r'\w+')
exwords = extokenizer.tokenize(ex.lower())


ex1 = np.unique(exwords)
print(len(ex1))
ex2 = dict((c, i) for i, c in enumerate(ex1))

print(len(ex2))

stop_words = set(stopwords.words('english'))

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text.lower())

unique_words = np.unique(words)
print(len(unique_words))
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

LENGTH = 5
next = []
prev = []
for j in range(len(words) - LENGTH):
     prev.append(words[j:j + LENGTH])
     next.append(words[j + LENGTH])
print(prev[0])
print(next[0])

X = np.zeros((len(prev), LENGTH, len(unique_words)), dtype=bool)
y = np.zeros((len(next), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev):
   for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
   y[i, unique_word_index[next[i]]] = 1

"""###Generate word vectors###"""

print(X.nbytes)
print(y.nbytes)

'''
Word2Vec 
'''
w2v = Word2Vec(full_t, size=300, window=10, min_count=1)
# train for more epochs
w2v.train(full_t, epochs=50, total_examples=len(full_t))
w2v_weights = w2v.wv.vectors

print(w2v.wv.vocab)
print(w2v.most_similar('the', topn=10))

"""###Create embed_matrix for each word###"""

vocab = w2v.wv.vocab
print(vocab)
vocab = list(vocab.keys())

word_vec_dict = {}
for word in vocab:
  word_vec_dict[word] = w2v.wv.get_vector(word)

embed_matrix = np.zeros(shape=(VOCAB_SIZE, EMBED_DIM))
for word, i in tok.word_index.items():
  embed_vector = word_vec_dict.get(word)
  if embed_vector is not None:
    embed_matrix[i] = embed_vector
  
print(embed_matrix)
print(len(embed_matrix))

"""###Model Architecture###"""

alpha_1 = .2
input_len = 1

model = Sequential()

model.add(LSTM(128, input_shape=(LENGTH, len(unique_words))))

model.add(Dropout(.1))

model.add(Dense(len(unique_words), activation=tf.nn.softmax))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(.01), metrics=['accuracy'])

model.summary()

save_checkpt = ModelCheckpoint(filepath='./autoc.h5', save_best_only=True, monitor='val_loss')

history = model.fit(X, y, epochs=1, batch_size=128, callbacks=[save_checkpt], validation_split=.05, shuffle=True)

model.save('next_sugg.h5')

acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()

# open file for writing
f = open("dict.txt","w")

# write file
f.write( str(unique_word_index) )

# close file
f.close()

import ast 
with open('dict.txt', 'r') as f:
  unique_word_index = f.read()
  unique_word_index = ast.literal_eval(unique_word_index)

  print(unique_word_index)

if "vk" not in unique_word_index.keys():
  print('ye')

# import pickle
# import heapq

model = load_model('next_sugg.h5')

model.summary()

msg = "Why is my project not"

text = msg.lower()

x = np.zeros((1, 5, len(unique_words) - 1))
for t, word in enumerate(text.split()):
    print(word)
    x[0, t, unique_word_index[word]] = 1

indices_word = dict((v, k) for k, v in unique_word_index.items())

pred = model.predict(x, verbose=0)[0]

i = np.argpartition(pred, -5)[-5:]

for indices in i:
  pred = indices_word[indices]

  print(pred)

msg += " " + pred

print(msg)



