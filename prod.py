
import pickle

import tensorflow as tf
import pandas as pd



##import the required libraries and APIs
##define tokenizing and padding parameters
vocab_size = 10000
max_length = 150
embedding_dim = 32
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000  # define training size(index)


from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model("lemmesaveyou.h5")
while(True):
    for i in range(16):
        print("\n")
    sentence = [input("Enter Sentence: ")]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    for i in model.predict(padded):
        for j in i:
            print(str(j*100)+"% chance of being a spam")
