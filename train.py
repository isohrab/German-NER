import conll_reader as cr
from config import Config
import gensim
import os
from model import Model


def get_data(config):
    test = cr.conllReader(config.test_filename)
    train = cr.conllReader(config.train_filename)

    # Build Word and Tag vocab
    if not (os.path.exists(config.words_filename) & os.path.exists(config.tags_filename)):
        vocab_words, vocab_tags = cr.get_vocabs([train, test])
        vocab_words.add(Config.UNK)
        vocab_words.add(Config.NUM)
        cr.write_vocab(vocab_tags, config.tags_filename)
        cr.write_vocab(vocab_words, config.words_filename)

    if not os.path.exists(config.chars_filename):
        # Build Char vocab
        vocab_chars = cr.get_char_vocab(train)
        cr.write_vocab(vocab_chars, config.chars_filename)
    # Get word2vec model from train data + test data
    # if os.path.exists(config.word2vec_filename):
    #     model = gensim.models.KeyedVectors.load_word2vec_format(config.word2vec_filename, binary=True)
    # else:
    #     sentences = cr.get_word2vec([test,train])
    #     model = gensim.models.Word2Vec(sentences, min_count=1, size=100, workers=4)
    #     model.wv.save_word2vec_format(config.word2vec_filename, binary=True)

    # load preprocessed vocabs
    try:
        vocab_words = cr.load_vocab(config.words_filename)
        vocab_tags  = cr.load_vocab(config.tags_filename)
        vocab_chars = cr.load_vocab(config.chars_filename)
    except IOError:
        print("Error loading words, tags, chars files")

    # Load wikipedia-200-mincount-20-window-8-cbow embedding file
    try:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(config.word2vec_filename, binary=True)
        embeddings = word2vec.syn0
    except IOError:
        print("error loading file with genism: wikipedia-200-mincount-20-window-8-cbow")

    # get processing functions
    processing_word = cr.get_processing_word(vocab_words, vocab_chars,
                    lowercase=False, chars=True)
    processing_tag  = cr.get_processing_word(vocab_tags,
                    lowercase=False)
    test = cr.conllReader(config.test_filename, processing_word, processing_tag)
    train = cr.conllReader(config.train_filename, processing_word, processing_tag)

    # Build Model
    model = Model(config, embeddings, len(vocab_tags), len(vocab_chars))

    model.build()

    model.train(train, test, vocab_tags)

if __name__ == "__main__":
    config = Config()
    get_data(config)
