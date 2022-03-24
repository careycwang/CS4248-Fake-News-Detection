import numpy as np

import nltk
from nltk import corpus


nltk.download("popular")


class GloVe2vec:
    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.embeddings_index = {}
        self._is_glove_embeddings_loaded = False


    def load_glove(self):
        embeddings_index = {}

        with open(self.glove_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        print(f"Found {len(embeddings_index)} word vectors.")

        self.embeddings_index = embeddings_index
        self._is_glove_embeddings_loaded = True


    def get_word_glove_vec(self, token, embedding_dim=300):
        """
        :parameter token: single token to be converted to GloVe vector
        :parameter embedding_dim: size of glove vector, default 300
        :returns vector of size embedding_dim
        """

        if not self._is_glove_embeddings_loaded:
            raise RuntimeError("Please load GloVe embeddings index first.")

        curr_vec = self.embeddings_index.get(token)

        if curr_vec is None:
            # token not found in embeddings, default to zero vector
            curr_vec = np.zeros((embedding_dim,))

        return curr_vec


    def get_glove_averaging_vec(
            self,
            tokens,
            embedding_dim=300,
            normalize=False,
            ignore_duplicates=False,
            ignore_stopwords=False,
    ):
        """
        :parameter tokens: iterable of tokens, e.g. ['i love cs', 'i hate statistics']
        :parameter embedding_dim: size of glove vector, default 300
        :parameter normalize: normalize averaged vectors to unit length
        :parameter ignore_duplicates: whether average vector takes into account of duplicate
            if True: duplicate vectors are summed
            if False: average over unique vectors only
        :parameter ignore_stopwords: whether vectors of stopwords are accounted for
        :returns vector of size embedding_dim
        """

        all_vecs = []
        seen_words = set()
        stopwords = set(corpus.stopwords.words("english"))

        for token in tokens:
            if ignore_duplicates and token in seen_words:
                continue

            if ignore_stopwords and token in stopwords:
                continue

            curr_vec = self.get_word_glove_vec(token, embedding_dim=embedding_dim)
            seen_words.add(token)

            all_vecs.append(curr_vec)

        if not all_vecs:
            all_vecs.append(np.zeros((embedding_dim,)))

        all_vecs = np.array(all_vecs)
        mean = all_vecs.mean(axis=0)

        if normalize:
            norm = np.linalg.norm(mean)
            if norm > 0:
                mean = mean / norm

        return mean


if __name__ == '__main__':
    model = GloVe2vec(glove_path='../data/glove.6B.300d.txt')
    model.load_glove()
    print(
        model.get_glove_averaging_vec('i love cs, i hate statistics'.split(), embedding_dim=300, normalize=True,
                                      ignore_duplicates=False, ignore_stopwords=False)
    )
    print(
        model.get_word_glove_vec("hello")
    )
