import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import nltk
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def train_model(model_name, x_train, y_train):
    ''' TODO: train your model based on the training data '''
    model = None
    if model_name == "nb":
        model = MultinomialNB()
    elif model_name == "lr":
        model = LogisticRegression(multi_class="multinomial", max_iter=2000)
    elif model_name == "svc":
        model = LinearSVC(max_iter=1000)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=0)

    model.fit(x_train, y_train)

    return model


def predict(model, x_test):
    ''' TODO: make your prediction here '''
    return model.predict(x_test)


def preprocess_sentence(sentence):
    wordnet = WordNetLemmatizer()
    # convert text to list of tokens
    words = nltk.word_tokenize(sentence)
    # lemmatize each word (token)
    return " ".join([wordnet.lemmatize(x, pos="v") for x in words])


def glove_embedding(embeddings_index, x_original):
    x_glove_list = []
    no_word_found_index = []

    for i in range(x_original.shape[0]):
        sentence = x_original[i]
        tokens = sentence.lower().split()
        vecs = []
        for word in tokens:
            try:
                vec = embeddings_index[word]
                vecs.append(vec)
            except KeyError:
                pass
        if len(vecs) > 0:
            vecs = np.array(vecs)
            x_glove_list.append(vecs.mean(axis=0))
        else:
            no_word_found_index.append(i)

    x_glove = np.zeros((len(x_glove_list), 300))
    for i in range(len(x_glove_list)):
        x_glove[i] = x_glove_list[i]

    return x_glove, no_word_found_index


def label_clean(no_word_found_index, y_original):
    y_clean = []

    for i in range(y_original.shape[0]):
        if i not in no_word_found_index:
            y_clean.append(y_original[i])

    return np.array(y_clean)


if __name__ == "__main__":
    np.random.seed(1)
    model_name = "lr"        # nb, lr, svc, rf
    vocab_dict = None
    maxlen = 700

    print("Model " + model_name + ":")

    print("start preprocess")
    train_csv = pd.read_csv("data/fulltrain.csv")
    y_train = train_csv.iloc[:, 0].to_numpy()
    x_train = train_csv.iloc[:, 1].to_numpy()
    test_csv = pd.read_csv("data/balancedtest.csv")
    y_test = test_csv.iloc[:, 0].to_numpy()
    x_test = test_csv.iloc[:, 1].to_numpy()

    for i in range(x_train.shape[0]):
        x_train[i] = preprocess_sentence(x_train[i].lower())
    for i in range(x_test.shape[0]):
        x_test[i] = preprocess_sentence(x_test[i].lower())

    print("start tokenize")

    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(x_train)
    #
    # sequences_train = tokenizer.texts_to_sequences(x_train)
    # x_train = pad_sequences(sequences_train, maxlen=maxlen)
    # sequences_test = tokenizer.texts_to_sequences(x_test)
    # x_test = pad_sequences(sequences_test, maxlen=maxlen)
    # vocab_dict = tokenizer.word_index

    embeddings_index = {}
    f = open("data/glove.6B.300d.txt", "r")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    x_train, train_no_word_found_index = glove_embedding(embeddings_index, x_train)
    y_train = label_clean(train_no_word_found_index, y_train)
    x_test, test_no_word_found_index = glove_embedding(embeddings_index, x_test)
    y_test = label_clean(test_no_word_found_index, y_test)

    print("start train")
    model = train_model(model_name, x_train, y_train)

    print("start predict")
    y_pred = predict(model, x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    print('f1-score on test = {}'.format(f1))
    print('accuracy on test = {}'.format(acc))
