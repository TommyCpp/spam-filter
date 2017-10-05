"""
Extra email based on word2vec result
"""
import pickle
import collections
import random

import numpy as np

embedding_size = 32


def read_data():
    with open("./data/source/dictionary.pkl", "rb") as f:
        dictionary = pickle.load(f)

    word_embeddings = np.genfromtxt('./data/source/word_embeddings.csv', delimiter=',')

    return dictionary, word_embeddings


def email2vec(content, dictionary, word_embeddings, n_words):
    words = content.split()
    words = list(filter(lambda x: x.isalpha() and len(x) > 1, words))
    vector = np.empty((0, embedding_size))
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    i = 0
    while i < n_words and i < len(dictionary.keys()) and i < len(count) - 1:
        if count[i][0] in dictionary.keys():
            vector = np.r_[vector, [word_embeddings[dictionary[count[i][0]]]]]
        i += 1

    if len(vector) < n_words:
        vector = np.r_[vector, [[0] * len(word_embeddings[0])] * (n_words - len(vector))]
    return vector


class Email:
    def __init__(self, subject, content, vector, is_spam):
        self.subject = subject
        self.content = content
        self.is_spam = is_spam
        self.vector = vector


SOURCE_MSG = "./data/source/msg.txt"
SOURCE_SPAM = "./data/source/spam.txt"


def extract_email(dictionary, word_embeddings, n_words) -> list:
    result = []
    with open(SOURCE_MSG) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            result.append(
                Email(lines[i], lines[i + 1], email2vec(lines[i + 1], dictionary, word_embeddings, n_words), 0))

    with open(SOURCE_SPAM) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            result.append(
                Email(lines[i], lines[i + 1], email2vec(lines[i + 1], dictionary, word_embeddings, n_words), 1))

    random.shuffle(result)  # shuffle the result
    return result


def batch(emails: list, batch_size):
    emails = random.shuffle(emails)
    for i in range(0, len(emails), batch_size):
        batch = emails[i, i + batch_size if batch_size > len(emails) else len(emails) - batch_size]
        label = [email.is_spam for email in batch]
        yield batch, label


N_WORDS = 16

dictionary, word_embeddings = read_data()

emails = extract_email(dictionary, word_embeddings, N_WORDS)

