import os
import string

import pandas as pd

import nltk
from nltk import word_tokenize, sent_tokenize, corpus
from nltk.stem import PorterStemmer, WordNetLemmatizer

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


nltk.download('popular')


class Doc2VecEmbedding:
    def __init__(self, vector_size=300, min_count=2, train_epoch=50):
        self.corpus = None
        self.model = None
        self.vector_size = vector_size
        self.min_count = min_count
        self.train_epoch = train_epoch


    def read_corpus(self, rows, is_processed=True):
        """
        Reads corpus from the dataframe

        :param rows: rows of documents or tokens
            if is_processed, expects 2D list, e.g. [['i', 'love', 'cs'], ['i', 'hate', 'statistics']]
            else, expects 1D list, e.g. ['i love cs', 'i hate statistics']
        :param is_processed: whether each row has been preprocessed, default True
        """
        my_corpus = []
        for i, row in enumerate(rows):
            if is_processed:
                tokens = row
            else:
                tokens = self.__tokenizer(row)

            my_corpus.append(TaggedDocument(tokens, [i]))

        self.corpus = my_corpus


    def train(self):
        """
        trains model based on corpus
        """
        if not self.corpus:
            raise RuntimeError('Please load corpus first.')

        self.model = Doc2Vec(vector_size=self.vector_size, min_count=self.min_count, epochs=self.train_epoch)
        self.model.build_vocab(self.corpus)


    def get_doc_vec(self, doc, is_processed=True):
        """
        :param doc: rows of documents or tokens
            if is_processed: expects list of tokens, e.g. ['i', 'love', 'cs']
            else: expects string, e.g. 'i love cs'
        :param is_processed: whether doc has been preprocessed, default True
        """
        if not self.corpus:
            raise RuntimeError('Please train model first.')

        if is_processed:
            tokens = doc
        else:
            tokens = self.__tokenizer(doc)

        return self.model.infer_vector(tokens)


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


if __name__ == '__main__':
    # path_to_df = os.path.join('..', 'data', 'raw_data', 'fulltrain.csv')

    path_to_df = './data/raw_data/fulltrain.csv'
    df = pd.read_csv(path_to_df, names=('Verdict', 'Text'))
    docs = df['Text'].values

    model = Doc2VecEmbedding()
    model.read_corpus(docs, is_processed=False)

    model.train()
    print(model.get_doc_vec('i love cs', is_processed=False))

