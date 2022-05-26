import json # for data
import pandas as pd # for data
import numpy as np # for data

import re # preprocessing
from nltk.corpus import stopwords # preprocessing
import nltk # preprocessing

import gensim.downloader as gensimAPI ## for word embedding

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
#for word embedding
from gensim.models import Word2Vec


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
    def fit(self, X, y):
            return self
    def transform(self, X):
            return np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def processText(text):
    ## removing the special characters (\r & \n) and punctuations, convert to lowercase, and strip
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()

    ## Tokenize (convert from string to list)
    wordTokens = text.split()

    ## remove Stopwords
    stopWords = set(stopwords.words('english'))
    filteredSentence = [w for w in wordTokens if not w in stopWords]

    ## Stemming (remove -ing, -ly, ...)
    ps = nltk.stem.porter.PorterStemmer()
    filteredSentence = [ps.stem(word) for word in filteredSentence]

    ## Lemmatisation (convert the word into root word)
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    filteredSentence = [lem.lemmatize(word) for word in filteredSentence]

    ## back to string from list
    text = " ".join(filteredSentence)
    return text


dictionary = []
with open('data.json', mode = 'r', errors = 'ignore') as json_file:
    for dic in json_file:
        dictionary.append( json.loads(dic) )
## print the first one
print(dictionary[0])
print()

## create dtf
dataSet = pd.DataFrame(dictionary)
## filter categories
dataSet = dataSet[ dataSet["category"].isin(['SCIENCE','WORLD NEWS','TECH']) ][["category","headline"]]
# dataSet = dataSet[ dataSet["category"].isin(['SPORTS','POLITICS','TECH']) ][["category","headline"]]

# print 5 random rows
# print(dataSet.sample(5))

dataSet["headline_clean"] = dataSet["headline"].apply(lambda x: processText(x))
# print(dataSet.head())

# split dataset
x_train, x_test, y_train, y_test = train_test_split(dataSet["headline_clean"],dataSet["category"],test_size=0.2,shuffle=True)

# # label encode the target variable
# encoder = preprocessing.LabelEncoder()
# y_train = encoder.fit_transform(y_train)
# y_test = encoder.fit_transform(y_test)


# Word embedding
dataSet["headline_clean_tokenized"]=[nltk.word_tokenize(i) for i in dataSet["headline_clean"]]
model = Word2Vec(dataSet["headline_clean_tokenized"], vector_size=300, window=8, min_count=5, sg=1, workers=30)
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))
modelw = MeanEmbeddingVectorizer(w2v)

x_train_tokenized = [nltk.word_tokenize(i) for i in x_train]
x_test_tokenized = [nltk.word_tokenize(i) for i in x_test]

x_train_vectors_w2v = modelw.transform(x_train_tokenized)
x_test_vectors_w2v = modelw.transform(x_test_tokenized)


# # Fitting the classification model using Logistic Regression
lr_w2v = LogisticRegression(solver='liblinear', C=10, penalty='l2')
lr_w2v.fit(x_train_vectors_w2v, y_train)  # model
# # Predict y value for test dataset
y_predict = lr_w2v.predict(x_test_vectors_w2v)
print(classification_report(y_test, y_predict))