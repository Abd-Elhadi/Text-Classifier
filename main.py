import json # for data
import pandas as pd # for data
import numpy as np # for data

import re # preprocessing
from nltk.corpus import stopwords # preprocessing
import nltk # preprocessing

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing

#for word embedding
from gensim.models import Word2Vec


def process_text(text):
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


def transform_w2v_to_vector(word2vec, vector_size, X):
    return np.array([
        np.mean([word2vec[w] for w in words if w in word2vec]
                or [np.zeros(vector_size)], axis=0)
        for words in X
    ])


dictionary = []
with open('data.json', mode = 'r', errors = 'ignore') as json_file:
    for dic in json_file:
        dictionary.append(json.loads(dic))


# create dtf
dataSet = pd.DataFrame(dictionary)
# filter categories
dataSet = dataSet[dataSet["category"].isin(['SCIENCE','WORLD NEWS','TECH'])][["category","headline"]]
# dataSet = dataSet[ dataSet["category"].isin(['SPORTS','POLITICS','TECH']) ][["category","headline"]]


dataSet["headline_clean"] = dataSet["headline"].apply(lambda x: process_text(x))

# split dataset
x_train, x_test, y_train, y_test = train_test_split(dataSet["headline_clean"],dataSet["category"],test_size=0.2,shuffle=True)

# label encode the target variable
# encoder = preprocessing.LabelEncoder()
# y_train = encoder.fit_transform(y_train)
# y_test = encoder.fit_transform(y_test)

# Word embedding
dataSet["headline_clean_tokenized"] = [nltk.word_tokenize(i) for i in dataSet["headline_clean"]]

vector_size = 100
model = Word2Vec(dataSet["headline_clean_tokenized"], vector_size=vector_size, window=8, min_count=5, sg=1, workers=10, epochs=8)
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))

x_train_tokenized = [nltk.word_tokenize(i) for i in x_train]
x_test_tokenized = [nltk.word_tokenize(i) for i in x_test]

x_train_vectors_w2v = transform_w2v_to_vector(w2v, vector_size, x_train_tokenized)
x_test_vectors_w2v = transform_w2v_to_vector(w2v, vector_size, x_test_tokenized)


# Fitting the classification model using logistic regression
print('\nLogistic Regression')
lr_model = LogisticRegression(solver='sag', C=8, multi_class='multinomial')
lr_model.fit(x_train_vectors_w2v, y_train)
# Predict y value for test dataset
y_predict_lr = lr_model.predict(x_test_vectors_w2v)
print(classification_report(y_test, y_predict_lr))
print('\n------------------------------------------\n')


# Fitting the classification model using svm
print('SVM')
svm_model = LinearSVC()
svm_model.fit(x_train_vectors_w2v, y_train)
# Predict y value for test dataset
y_predict_svm = svm_model.predict(x_test_vectors_w2v)
print(classification_report(y_test, y_predict_svm))
print('\n------------------------------------------\n')


# Fitting the classification model using random forrest
print('Random Forrest')
rf_model = RandomForestClassifier()
rf_model.fit(x_train_vectors_w2v, y_train)
# Predict y value for test dataset
y_predict_rf = rf_model.predict(x_test_vectors_w2v)
print(classification_report(y_test, y_predict_rf))


