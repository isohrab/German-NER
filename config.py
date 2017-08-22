import os

class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

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
    char_embedding_dim = 100
    hidden_size = 100
    filter_sizes = [1, 2, 3, 4, 5, 6]
    num_filters = 32
    batch_size = 60
    max_length_word = 30
    nepochs = 1
    lr = 0.001
    lr_decay = 0.
    dropout = 0.5

    # model config
    crf = False

