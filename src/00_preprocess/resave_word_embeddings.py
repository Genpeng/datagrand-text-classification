# _*_ coding: utf-8 _*_

"""
Re-save the file of word embeddings to `npy` format and
construct the mapping between words and its corresponding indices.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/07/29
"""

import pickle
import numpy as np
import pandas as pd

EMBEDDING_SIZE = 300
SPECIAL_SYMBOL = ['<PAD>', '<UNK>']

# Load words and its corresponding embeddings
print("Load words and its corresponding embeddings...")
word_embedding_file = "../../word_vectors/gemsim-word-300d-mc5.txt"
with open(word_embedding_file, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()[1:]
    words = list(SPECIAL_SYMBOL)
    word_embeddings = list()
    word_embeddings.append(np.zeros(EMBEDDING_SIZE, dtype=np.float32))  # the values of 'PAD' are all zero
    word_embeddings.append(np.random.randn(EMBEDDING_SIZE))  # the values of 'UNK' satisfy the normal distribution
    for line in lines:
        cols = line.split()
        words.append(cols[0])
        word_embeddings.append(np.array(cols[1:], dtype=np.float32))

id2word_series = pd.Series(words, index=range(len(words)))
word2id_series = pd.Series(range(len(words)), index=words)
word_embeddings = np.vstack(word_embeddings)

# Save to file
print("Save to file...")
id2word_file = "../../processed_data/id2word.pkl"
word2id_file = "../../processed_data/word2id.pkl"
word_embeddings_file = "../../word_vectors/word-embedding-300d-mc5.npy"
with open(id2word_file, 'wb') as fout:
    pickle.dump(id2word_series, fout)
with open(word2id_file, 'wb') as fout:
    pickle.dump(word2id_series, fout)
np.save(word_embeddings_file, word_embeddings)
print("Finished! ( ^ _ ^ ) V")
