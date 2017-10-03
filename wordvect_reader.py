import pickle
import collections
import tensorflow as tf
import numpy as np

"""
Use to read word2vec result from the file.
"""

with open("./data/source/dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)

# print(dictionary)

word_embeddings = np.genfromtxt('./data/source/word_embeddings.csv', delimiter=',')


# print(word_embeddings)


# Test hit rate
class Email:
    def __init__(self, subject, content, is_spam):
        self.subject = subject
        self.content = content
        self.is_spam = is_spam


SOURCE_MSG = "./data/source/msg.txt"
SOURCE_SPAM = "./data/source/spam.txt"


def extract_email() -> list:
    result = []
    with open(SOURCE_MSG) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            result.append(Email(lines[i], lines[i + 1], False))

    with open(SOURCE_SPAM) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            result.append(Email(lines[i], lines[i + 1], True))

    return result


def read_data(emails: list):
    data = []
    for email in emails:
        data.extend(email.content.split())
    return data


vocabulary = read_data(extract_email())
print("Data size", len(vocabulary))


def build_dataset(words, n_words, dictionary):
    words = list(filter(lambda x: x.isalpha() and len(x) > 1, words))
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    i = 0
    while i < n_words and i < len(dictionary.keys()):
        if count[i][0] in dictionary.keys():
            print(count[i][0])
        i += 1
    return i


print(build_dataset(vocabulary, 60000, dictionary))
