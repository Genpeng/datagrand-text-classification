# _*_ coding: utf-8 _*_

"""
Train distributed representation of words by using gensim. Deprecated!!!

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/07/25
"""

import os
from time import time
from gensim.models import Word2Vec


class SentenceIterator:
    """Sentence iterator, sequentially return a sentence."""

    def __init__(self, dirname, col=0):
        self.dirname = dirname
        self.col = col

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            i = 0
            for line in open(os.path.join(self.dirname, fname)):
                if i == 0:  # skip the header
                    continue
                else:
                    i += 1
                    sentence = line.split(',')[self.col]
                    yield sentence.split()


if __name__ == '__main__':
    corpus_path = "../../test_data/"
    sentences = SentenceIterator(corpus_path, col=2)

    print("Start training...")
    t0 = time()
    model = Word2Vec(sentences, size=300, min_count=1, sg=0, iter=30, workers=16, seed=42)
    print("Done in %.3f second!" % (time() - t0))
    print("Training finished! ( ^ _ ^ ) V")

    print("Save to file...")
    model.wv.save("gensim-word-300d.bin")
    model.wv.save_word2vec_format("gensim-word-300d.txt", binary=False)