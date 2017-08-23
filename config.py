import os

class DefaultConfig():
    # def __init__(self):
    #     # directory for training outputs
    #     if not os.path.exists(self.output_path):
    #         os.makedirs(self.output_path)

    # Path config
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    # dataset
    test_filename = "data/test_data"
    train_filename = "data/train_data"
    #word2vec_filename = "data/wikipedia-200-mincount-20-window-8-cbow.bin"
    word2vec_filename = "data/wikipedia-100-mincount-20-window-5-cbow.bin"
    tags_filename = "data/tags.txt"
    words_filename = "data/words.txt"
    chars_filename = "data/chars.txt"
    UNK = "$UNK$"
    NUM = "$NUM$"
    NONE = "O"

    # Hyper parameters
    CHAR_EMB_DIM = 100
    HIDDEN_SIZE = 100
    FILTER_SIZE = [1, 2, 3, 4, 5, 6]
    N_FILTERS = 32
    BATCH_SIZE = 60
    MAX_LENGTH_WORD = 45
    N_EPOCHS = 15
    LR = 0.001
    LR_DECAY = 0.9
    DROPOUT = 0.5


