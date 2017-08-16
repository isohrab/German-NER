import conll_reader as cr
from config import Config
import gensim
import os


def build_data(config):
    test = cr.conllReader(config.test_filename)
    train = cr.conllReader(config.train_filename)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = cr.get_vocabs([train, test])

    # Build Char vocab
    vocab_char = cr.get_char_vocab(train)

    # Get word2vec model from train data + test data
    if os.path.exists(config.word2vec_filename):
        model = gensim.models.KeyedVectors.load_word2vec_format(config.word2vec_filename, binary=True)
    else:
        sentences = cr.get_word2vec([test,train])
        model = gensim.models.Word2Vec(sentences, min_count=1, size=100, workers=4)
        model.wv.save_word2vec_format(config.word2vec_filename, binary=True)


if __name__ == "__main__":
    config = Config()
    build_data(config)
