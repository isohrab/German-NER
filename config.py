import os

class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # create instance of logger
        #self.logger = get_logger(self.log_path)


    # Path config
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    # dataset
    test_filename = "data/test_data"
    train_filename = "data/train_data"
    word2vec_filename = "data/wikipedia-200-mincount-20-window-8-cbow.bin"
    tags_filename = "data/tags.txt"
    words_filename = "data/words.txt"
    chars_filename = "data/chars.txt"

    # Hyper parameters
    char_dim = 100
    hidden_size = 100
    filter_sizes = [3, 4, 5]
    num_filters = 120
    batch_size = 60

    # model config
    crf = False

