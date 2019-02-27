# coding: utf-8
# @author: Shaw
# @datetime: 2019-02-27 16:45
# @Name: movie_classify.py

import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

if __name__ == "__main__":
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    word_index = {k: v + 3 for k, v in word_index.item()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    decode_review(train_data[0])
