import pickle
import tensorflow as tf
import numpy as np


with open("./data/source/dictionary.pkl","rb") as f:
    dictionary = pickle.load(f)

word_embeddings = np.genfromtxt('./data/source/word_embeddings.csv', delimiter=',')

print(word_embeddings)
