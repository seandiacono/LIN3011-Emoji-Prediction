import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


np.random.seed(500)

with open('data/dataset/train/us_train.text', 'r', encoding="utf8") as file:
    X_train1 = file.readlines()

with open('data/dataset/train/us_train.labels', 'r', encoding="utf8") as file:
    y_train1 = file.readlines()

with open('data/dataset/test/us_test.text', 'r', encoding="utf8") as file:
    X_test1 = file.readlines()

with open('data/dataset/test/us_test.labels', 'r', encoding="utf8") as file:
    y_test = file.readlines()


X_train1 = [tweet.lower() for tweet in X_train1]
X_train1 = [word_tokenize(tweet) for tweet in X_train1]
X_test1 = [tweet.lower() for tweet in X_test1]
X_test1 = [word_tokenize(tweet) for tweet in X_test1]

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

X_train = []

for tweet in X_train1:
    words = []
    word_lemmatizer = WordNetLemmatizer()

    for word in pos_tag(tweet):
        print(word)
        if word[0] not in stopwords.words('english') and word[0].isalpha():
            word = word_lemmatizer.lemmatize(word[0], tag_map[word[1]])
            words.append(word)

    X_train.append(words)

print(X_train[90])
