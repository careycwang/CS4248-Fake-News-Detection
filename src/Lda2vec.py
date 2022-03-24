import os
import string

import numpy as np
import pandas as pd

import nltk
from nltk import word_tokenize, sent_tokenize, corpus
from nltk.stem import PorterStemmer, WordNetLemmatizer

import gensim
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary


nltk.download('popular')


class Lda2vec:
    def __init__(self, tokenizer=None, num_topics=50):
        '''
        :parameter tokenizer: tokenizer function, default nltk word tokenizer
        :parameter num_topics: number of topics, default 50
        '''
        self.num_topics = num_topics
        self.tokenizer = tokenizer if tokenizer is not None else self.__tokenizer

        self.dictionary = None
        self.lda = None
        self.__is_fitted = False


    def __check_fitted(self):
        if not self.__is_fitted:
            raise RuntimeError('Model has not been fitted, call fit(input_corpus) first.')


    def __tokenizer(self, text):
        tokens = []
        porter_stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(corpus.stopwords.words('english'))

        text = ''.join(char for char in text if char not in string.punctuation)

        for sent in sent_tokenize(text, language='english'):
            for word in word_tokenize(sent, language='english'):
                if len(word) < 2 or word.lower() in stop_words:
                    continue

                word = lemmatizer.lemmatize(word)
                # word = porter_stemmer.stem(word)
                tokens.append(word)

        return tokens


    def fit(self, input_tokens=None, input_strings=None):
        '''
        :parameter input_strings: iterable of strings,
            e.g. ['i love cs', 'i hate statistics']
        :parameter input_tokens: iterable of iterable of tokens,
            e.g. [['i', 'love', 'cs'], ['i', 'hate', 'statistics']]
        '''

        if input_strings is None and input_tokens is None:
            raise RuntimeError('Either input_tokens or input_strings must not be None.')

        if not input_tokens:
            input_tokens = list(map(self.tokenizer, input_strings))

        self.dictionary = Dictionary(input_tokens)
        self.corpus = [self.dictionary.doc2bow(tokens) for tokens in input_tokens]
        self.lda = LdaModel(self.corpus, num_topics=self.num_topics, alpha='auto', eval_every=5)
        self.__is_fitted = True


    def get_doc_vec(self, words):
        '''
        :parameter words: iterable of tokens, e.g. ['i', 'love', 'cs']
        :returns np.ndarray of shape (num_topics, ) where each value represents the prob of the words being in that topic
        '''
        self.__check_fitted()
        vec = np.zeros(self.num_topics, )
        bow = self.dictionary.doc2bow(words)
        for i, p in self.lda[bow]:
            vec[i] = p
        return vec


if __name__ == '__main__':
    data_dir = '../data/'
    df = pd.read_csv(os.path.join(data_dir, 'fulltrain.csv'), names=('Verdict', 'Text'))

    model = Lda2vec(num_topics=10)
    model.fit(input_strings=df['Text'])
    x = model.get_doc_vec('i love cs'.split())
