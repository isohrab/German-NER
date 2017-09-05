import os

class DefaultConfig():
    ###     Path     ###
    #   graph Path
    output_path = "results/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "logs/"
    # preprocessed Data path
    test_filename = "data/test_data"
    train_filename = "data/train_data"
    #word2vec_filename = "data/wikipedia-200-mincount-20-window-8-cbow.bin"
    word2vec_filename = "data/wikipedia-100-mincount-30-window-8-cbow.bin"
    # preprocessed Data file names
    tags_filename = "data/tags.txt"
    words_filename = "data/words.txt"
    chars_filename = "data/chars.txt"
    # preprocessed data variables
    UNK = "$UNK$"
    NUM = "$NUM$"
    NONE = "O"

    ###  Hyper parameters    ###
    BATCH_SIZE = 40
    MAX_LENGTH_WORD = 50
    N_EPOCHS = 100
    LR = 0.001
    LR_DECAY = 0.95
    DROPOUT = 0.75
    # Char Embedding (CNN)
    CHAR_EMB_DIM = 120
    FILTER_SIZE = [2, 3, 4, 5]
    N_FILTERS = 96
    # BiLSTM
    HIDDEN_SIZE = 400


