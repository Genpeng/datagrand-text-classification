
# 1. Train word embeddings
# python train_word_embeddings.py
nohup python -u train_word_embeddings.py > train_word_embeddings.log 2>&1 &

# 2. Train char embeddings
# python train_char_embeddings.py
nohup python -u train_char_embeddings.py > train_char_embeddings.log 2>&1 &

# 3. Re-save the file of word embeddings to `npy` format
python resave_embeddings.py

# 4. Transform the words (or characters) of all the samples to its ids
python transform_data_text.py

# 5. Split training training set and validation set into small batches
python create_batch_data.py