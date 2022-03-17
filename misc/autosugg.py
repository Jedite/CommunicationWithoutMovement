# preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils.np_utils as ku
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize
from gensim.models import Word2Vec


# model architecture
from keras.models import load_model, Sequential
from keras.layers import Bidirectional, LSTM, Embedding, Dense, Dropout, Flatten, Input, CuDNNLSTM, Conv1D, MaxPooling1D, LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2
from keras import Model
import keras
import tensorflow as tf

# other + standard libraries
import matplotlib.pyplot as plt
import numpy as np
import string
import re
import tarfile
import gc
import os
import sys
import pickle

gc.enable()

# download necessary stop words
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# glob variables
MAX_SAMPLES = 10000
EMBED_DIM = 300
LATENT_DIM = 512
NUM_HEADS = 8
MAX_LENGTH = 40
EPOCHS = 100


# load data

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


# preprocessing

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

# just to check memory
print(X.nbytes)
print(y.nbytes)


# model architecture
model = Sequential()

model.add(LSTM(128, input_shape=(LENGTH, len(unique_words))))

model.add(Dropout(.1))

model.add(Dense(len(unique_words), activation=tf.nn.softmax))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(.01), metrics=['accuracy'])

model.summary()

# Train model, change epoch amount
save_checkpt = ModelCheckpoint(filepath='./auto.h5', save_best_only=True, monitor='val_loss')

history = model.fit(X, y, epochs=EPOCHS, batch_size=128, callbacks=[save_checkpt], validation_split=.05, shuffle=True)

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


# prediction

model = load_model('next_sugg.h5')

with open('indices_char.p', 'rb') as f:
  indices_char = pickle.load(f)

def prepare_input(text):
    # print(len(text.split()))
    text = text.replace(',', '')
    x = np.zeros((1, len(text.split()), len(unique_words)))
    for t, word in enumerate(text.split()):
        print(word)
        x[0, t, unique_word_index[word]] = 1
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion

def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]   

quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]

for q in quotes:
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()