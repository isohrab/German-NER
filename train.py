import conll_reader as cr
from config import DefaultConfig
import gensim
import os
import sys
import tensorflow as tf
from model import Model
from data_helper import batch_gen


def train_model(cfg, train_set, dev_set, embeddings, tags, chars):

    # Build Model
    model = Model(cfg, embeddings, len(tags), len(chars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(cfg.N_EPOCHS):

            train_losses = 0.0
            validation_loss = 0.0
            accuracy = 0.0
            for words, labels in batch_gen(train_set, cfg.BATCH_SIZE):
                fd, _ = model.get_feed_dict(words, labels, cfg.LR, cfg.DROPOUT)
                _, train_loss = sess.run([model.train_op, model.loss], feed_dict=fd)
                train_losses += train_loss

            accuracy, f1, validation_loss = model.run_evaluate(sess, dev_set, tags)
            # decay learning rate
            cfg.LR *= cfg.LR_DECAY

            print("epoch %d - train loss: %.2f, validation loss: %.2f, accuracy: %.2f width f1: %.2f" % \
                (epoch + 1, train_losses, validation_loss, accuracy * 100, f1))


if __name__ == "__main__":
    # TODO: we MUST use embedding as argument
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: %s TRAIN_SET DEV_SET\n" % sys.argv[0])
        sys.exit(1)

    cfg = DefaultConfig()

    # check if data not processed, generate tags, words, chars
    if not (os.path.exists(cfg.words_filename) & os.path.exists(cfg.tags_filename) & os.path.exists(cfg.chars_filename)):
        train = cr.conllReader(sys.argv[2])
        test = cr.conllReader(sys.argv[3])

        vocab_words, vocab_tags = cr.get_vocabs([train, test])
        vocab_words.add(cfg.UNK)
        vocab_words.add(cfg.NUM)
        cr.write_vocab(vocab_tags, cfg.tags_filename)
        cr.write_vocab(vocab_words, cfg.words_filename)

        vocab_chars = cr.get_char_vocab(train)
        cr.write_vocab(vocab_chars, cfg.chars_filename)

    # load preprocessed vocabs
    try:
        vocab_words = cr.load_vocab(cfg.words_filename)
        vocab_tags  = cr.load_vocab(cfg.tags_filename)
        vocab_chars = cr.load_vocab(cfg.chars_filename)
    except IOError:
        print("Error loading words, tags, chars files")

    # Load wikipedia-200-mincount-20-window-8-cbow embedding file
    try:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)
        embeddings = word2vec.syn0
    except IOError:
        print("error loading file with genism: wikipedia-200-mincount-20-window-8-cbow")

    # get processing functions
    processing_word = cr.get_processing_word(vocab_words, vocab_chars,
                    lowercase=False, chars=True)
    processing_tag  = cr.get_processing_word(vocab_tags,
                    lowercase=False)

    train = cr.conllReader(sys.argv[2], processing_word, processing_tag)
    test = cr.conllReader(sys.argv[3], processing_word, processing_tag)

    train_model(cfg, train, test,  embeddings, vocab_tags, vocab_chars)
