# _*_ coding: utf-8 _*_

"""
Train distributed representation of words using gensim.

Author: StrongXGP
Date:	2018/07/12
"""

from gensim.models.word2vec import Word2Vec

# Load data
# =========================================================================

print("Loading data...")

train_data_file = "../raw_data/train_set.csv"
test_data_file = "../raw_data/test_set.csv"

train_lines = open(train_data_file, 'r', encoding='utf-8').read().splitlines()[1:]
test_lines = open(test_data_file, 'r', encoding='utf-8').read().splitlines()[1:]

train_char_samples = [line.split(',')[1] for line in train_lines]
test_char_samples = [line.split(',')[1] for line in test_lines]
char_samples = train_char_samples + test_char_samples

char_samples = [char_sample.split() for char_sample in char_samples]

# Train and save word2vec model
# =========================================================================

print("Start training...")

model = Word2Vec(char_samples, size=300)
model.wv.save("datagrand-char-300d.bin")
model.wv.save_word2vec_format("datagrand-char-300d.txt", binary=False)

print("Training Finish! ^_^")