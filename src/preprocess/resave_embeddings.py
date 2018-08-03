# _*_ coding: utf-8 _*_

"""
Re-save the file of word (character) embeddings to `npy` format and
construct the mapping between words and its corresponding indices.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/07/29
"""

import gc
import pickle
import numpy as np

EMBEDDING_SIZE = 300
SPECIAL_SYMBOLS = ['<PAD>', '<UNK>']


def load_embedding(embedding_file):
    """Load embeddings from file."""
    np.random.seed(42)

    with open(embedding_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()[1:]  # remove first line (embedding description line)

    id_to_symbol_map = dict()
    symbol_to_id_map = dict()
    for i, symbol in enumerate(SPECIAL_SYMBOLS):
        id_to_symbol_map[i] = symbol
        symbol_to_id_map[symbol] = i

    num_total_symbols = len(SPECIAL_SYMBOLS) + len(lines)
    embeddings = np.zeros((num_total_symbols, EMBEDDING_SIZE), dtype=np.float32)
    embeddings[1] = np.random.randn(EMBEDDING_SIZE)  # the values of 'UNK' satisfy the normal distribution

    index = 2
    for line in lines:
        cols = line.split()
        id_to_symbol_map[index] = cols[0]
        symbol_to_id_map[cols[0]] = index
        embeddings[index] = np.array(cols[1:], dtype=np.float32)
        index += 1

    return id_to_symbol_map, symbol_to_id_map, embeddings


def main():
    print("[INFO] Load character embeddings...")
    char_embedding_file = "../../embeddings/datagrand-char-300d.txt"
    id_to_char_map, char_to_id_map, char_embeddings = load_embedding(char_embedding_file)
    print("[INFO] Finished!")

    print("[INFO] Save character embeddings...")
    id_to_char_file = "../../embeddings/id2char.pkl"
    char_to_id_file = "../../embeddings/char2id.pkl"
    char_embedding_resave_file = "../../embeddings/char-embedding-300d.npy"
    with open(id_to_char_file, 'wb') as fout:
        pickle.dump(id_to_char_map, fout)
    with open(char_to_id_file, 'wb') as fout:
        pickle.dump(char_to_id_map, fout)
    np.save(char_embedding_resave_file, char_embeddings)
    print("[INFO] Finish!")

    del id_to_char_map, char_to_id_map, char_embeddings
    gc.collect()

    print("[INFO] Load word embeddings...")
    word_embedding_file = "../../embeddings/datagrand-word-300d-mc5.txt"
    id_to_word_map, word_to_id_map, word_embeddings = load_embedding(word_embedding_file)
    print("[INFO] Finished!")

    print("[INFO] Save word embeddings...")
    id_to_word_file = "../../embeddings/id2word.pkl"
    word_to_id_file = "../../embeddings/word2id.pkl"
    word_embedding_resave_file = "../../embeddings/word-embedding-300d-mc5.npy"
    with open(id_to_word_file, 'wb') as fout:
        pickle.dump(id_to_word_map, fout)
    with open(word_to_id_file, 'wb') as fout:
        pickle.dump(word_to_id_map, fout)
    np.save(word_embedding_resave_file, word_embeddings)
    print("[INFO] Finished!")


if __name__ == '__main__':
    main()
