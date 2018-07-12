# _*_ coding: utf-8 _*_

"""
Train distributed representation of words using gensim.

Author: StrongXGP
Date:	2018/07/12
"""

import gc
from gensim.models.word2vec import Word2Vec

# Load data
# =========================================================================

print("Loading data...")

data_file = "../raw_data/demo.csv"
lines = open(data_file, 'r', encoding='utf-8').read().splitlines()[1:]
char_samples = [line.split(',')[1] for line in lines]
char_samples = [char_sample.split() for char_sample in char_samples]

del lines
gc.collect()

# Train and save word2vec model
# =========================================================================

print("Start training...")

model = Word2Vec(char_samples, size=300, min_count=1)
model.wv.save("datagrand-char-300d.bin")
model.wv.save_word2vec_format("datagrand-char-300d.txt", binary=False)

print("Training Finish! ^_^")