import conll_reader as cr
from config import DefaultConfig
import gensim
import os
import sys
import tensorflow as tf
from model import Model
from data_helper import batch_gen


def train_model(cfg, train_set, dev_set, embed, tags, chars):

    # Build Model
    model = Model(cfg, embed, len(tags), len(chars))
    # initial session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # create log writer
        summary_writer = tf.summary.FileWriter(cfg.log_path, graph=tf.get_default_graph())
        # run epoch
        for epoch in range(cfg.N_EPOCHS):
            train_losses = 0.0
            validation_loss = 0.0
            accuracy = 0.0
            # Run batches
            i = 0 # counter for summary results.
            for words, labels in batch_gen(train_set, cfg.BATCH_SIZE):
                fd, _ = model.get_feed_dict(words, labels, cfg.LR, cfg.DROPOUT)
                # train model
                _, train_loss, summary = sess.run([model.train_op, model.loss, model.merged_summary_op], feed_dict=fd)
                train_losses += train_loss
                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * cfg.BATCH_SIZE + i)
                i += 1
            # Evaluate model after training
            accuracy, f1, validation_loss, p, r = model.run_evaluate(sess, dev_set, tags)
            # decay learning rate
            cfg.LR *= cfg.LR_DECAY

            print("epoch %d - train loss: %.2f, validation loss: %.2f, accuracy: %.2f with f1: %.2f, P: %.2f, R: %.2f" % \
                (epoch + 1, train_losses, validation_loss, accuracy * 100, f1 * 100, p * 100, r * 100))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: %s wikipedia-xxx-mincount-xx-window-x-cbow.bin TRAIN_SET DEV_SET\n" % sys.argv[0])
        sys.exit(1)

    cfg = DefaultConfig()

    # check if data not processed, generate tags, words, chars
    if not (os.path.exists(cfg.words_filename) & os.path.exists(cfg.tags_filename) & os.path.exists(cfg.chars_filename)):
        print("preprocessed Data not found. processing data...")
        train = cr.conllReader(sys.argv[2])
        test = cr.conllReader(sys.argv[3])

        # get words and tags vocabulary from whole data
        vocab_words, vocab_tags = cr.get_vocabs([train, test])
        # Add unknown token and number to vocab
        vocab_words.add(cfg.UNK)
        vocab_words.add(cfg.NUM)
        # save all words and tags to file
        cr.write_vocab(vocab_tags, cfg.tags_filename)
        cr.write_vocab(vocab_words, cfg.words_filename)
        # get and save chars from dataset to file
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
    # Load wikipedia-xxx-mincount-xx-window-x-cbow embedding file
    try:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)
        embeddings = word2vec.syn0
    except IOError:
        print("error loading file with genism: wikipedia-200-mincount-20-window-8-cbow")

    # assign processing options to processing function
    processing_word = cr.get_processing_word(vocab_words, vocab_chars,
                    lowercase=False, chars=True)
    processing_tag  = cr.get_processing_word(vocab_tags,
                    lowercase=False)
    # read trian and test set
    train = cr.conllReader(sys.argv[2], processing_word, processing_tag)
    test = cr.conllReader(sys.argv[3], processing_word, processing_tag)
    # train and test model
    train_model(cfg, train, test,  embeddings, vocab_tags, vocab_chars)
