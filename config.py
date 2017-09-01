import os

class DefaultConfig():
    # def __init__(self):
    #     # directory for training outputs
    #     if not os.path.exists(self.output_path):
    #         os.makedirs(self.output_path)

    # Path config
    output_path = "results/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    # dataset
    test_filename = "data/test_data"
    train_filename = "data/train_data"
    #word2vec_filename = "data/wikipedia-200-mincount-20-window-8-cbow.bin"
    word2vec_filename = "data/wikipedia-100-mincount-30-window-8-cbow.bin"
    tags_filename = "data/tags.txt"
    words_filename = "data/words.txt"
    chars_filename = "data/chars.txt"
    UNK = "$UNK$"
    NUM = "$NUM$"
    NONE = "O"

    # Hyper parameters
    CHAR_EMB_DIM = 120
    HIDDEN_SIZE = 400
    FILTER_SIZE = [2, 3, 4]
    N_FILTERS = 64
    BATCH_SIZE = 40
    MAX_LENGTH_WORD = 50
    N_EPOCHS = 100
    LR = 0.001
    LR_DECAY = 0.95
    DROPOUT = 0.5

    CRF = False


