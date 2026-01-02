from gensim.models import KeyedVectors, word2vec

import numpy as np
import pandas as pd
import re
import string
import dill
import pickle
import os

from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords as sw, wordnet as wn

from sklearn.base import BaseEstimator, TransformerMixin

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import (
    Dense, LSTM, SpatialDropout1D, Activation, Conv1D,
    MaxPooling1D, Input, Embedding, BatchNormalization, concatenate
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using NLTK tokenization, POS tagging, lemmatization and vectorization.
    """

    def __init__(self, max_sentence_len=300, stopwords=None, punct=None, lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.max_sentence_len = max_sentence_len

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return X

    def transform(self, X):
        return [self.tokenize(doc) for doc in X]

    def tokenize(self, document):
        lemmatized_tokens = []

        # Basic text normalization
        document = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", document)
        document = re.sub(r"what's", "what is ", document)
        document = re.sub(r"\'s", " ", document)
        document = re.sub(r"\'ve", " have ", document)
        document = re.sub(r"can't", "cannot ", document)
        document = re.sub(r"n't", " not ", document)
        document = re.sub(r"i'm", "i am ", document)
        document = re.sub(r"\'re", " are ", document)
        document = re.sub(r"\'d", " would ", document)
        document = re.sub(r"\'ll", " will ", document)
        document = re.sub(r"(\d+)(k)", r"\g<1>000", document)

        # Sentence tokenization
        for sent in sent_tokenize(document):
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                token = token.lower() if self.lower else token
                if self.strip:
                    token = token.strip().strip('_').strip('*')

                if token in self.stopwords or all(char in self.punct for char in token):
                    continue

                lemma = self.lemmatize(token, tag)
                lemmatized_tokens.append(lemma)

        return ' '.join(lemmatized_tokens)

    def vectorize(self, doc):
        """
        Returns a vectorized padded version of sequences.
        """
        save_path = "Models/padding.pickle"
        with open(save_path, 'rb') as f:
            tokenizer = pickle.load(f)

        doc_pad = tokenizer.texts_to_sequences(doc)
        return np.squeeze(doc_pad)

    def lemmatize(self, token, tag):
        tag_map = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }
        return self.lemmatizer.lemmatize(token, tag_map.get(tag[0], wn.NOUN))
