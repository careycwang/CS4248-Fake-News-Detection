import os

import numpy as np
import pandas as pd

import lda2vec
from lda2vec import model, utils
from lda2vec.nlppipe import Preprocessor


def main():
    base_dir = ''
    data_dir = os.path.join(base_dir, 'data')
    raw_data_dir = os.path.join(data_dir, 'raw_data')
    clean_data_dir = os.path.join(data_dir, 'test')

    open(os.path.join(raw_data_dir, 'fulltrain.csv'), 'r').close()

    glove_embedding_path = os.path.join(base_dir, 'data', 'glove.6B.300d.txt')

    df = pd.read_csv(os.path.join(raw_data_dir, 'fulltrain.csv'), names=('Verdict', 'Text'))

    prep = Preprocessor(df, "Text", max_features=3000, maxlen=1000, min_count=20)

    prep.preprocess()
    embedding_matrix = prep.load_glove(glove_embedding_path)

    prep.save_data(clean_data_dir, embedding_matrix=embedding_matrix)

    idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids = utils.load_preprocessed_data(clean_data_dir,
                                                                                                   load_embed_matrix=False)

    # Number of unique documents
    num_docs = doc_ids.max() + 1
    # Number of unique words in vocabulary (int)
    vocab_size = len(freqs)
    # Embed layer dimension size
    # If not loading embeds, change 128 to whatever size you want.
    embed_size = embedding_matrix.shape[1]
    # Number of topics to cluster into
    num_topics = 20
    # Amount of iterations over entire dataset
    num_epochs = 200
    # Batch size - Increase/decrease depending on memory usage
    batch_size = 4096
    # Epoch that we want to "switch on" LDA loss
    switch_loss_epoch = 0
    # Pretrained embeddings value
    pretrained_embeddings = embedding_matrix
    # If True, save logdir, otherwise don't
    save_graph = True

    # Initialize the model
    m = model(num_docs,
              vocab_size,
              num_topics,
              embedding_size=embed_size,
              pretrained_embeddings=pretrained_embeddings,
              freqs=freqs,
              batch_size=batch_size,
              save_graph_def=save_graph)

    # Train the model
    m.train(pivot_ids,
            target_ids,
            doc_ids,
            len(pivot_ids),
            num_epochs,
            idx_to_word=idx_to_word,
            switch_loss_epoch=switch_loss_epoch)


if __name__ == '__main__':
    main()
