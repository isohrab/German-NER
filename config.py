import os

class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # create instance of logger
        #self.logger = get_logger(self.log_path)


    # general config
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    # dataset
    test_filename = "data/test_data"
    train_filename = "data/train_data"
    word2vec_filename = "data/word2vec"
