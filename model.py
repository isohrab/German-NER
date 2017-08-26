import tensorflow as tf
from data_helper import pad_sequences, batch_gen, get_chunks
import numpy as np


class Model(object):
    def __init__(self, config, embeddings, ntags, nchars):
        '''
        Tensorflow model
        :param embeddings: word2vec embedding file which produced by gensim
        :param ntags: number of tags
        :param nchars: number of chars
        '''
        self.cfg = config
        self.embeddings = embeddings
        self.nchars = nchars
        self.ntags = ntags

        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()
        self.add_train_op()

    def add_placeholders(self):
        '''
        add placeholder to self
        '''
        # Shape = (batch size, max length of sentences in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")

        # Shape = (batch size)
        self.sentences_lengths = tf.placeholder(tf.int32, shape=[None], name="sentences_lengths")

        # Shape = (batch size, max length of sentences, max length of words)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")

        # Shape = (batch size, max length of sentences)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_length")

        # Shape = (batch size, max length of sentences)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

        # Learning rate for Optimization
        self.lr = tf.placeholder(tf.float32, shape=[], name="Learning_rate")

        # Dropout
        self.dropout = tf.placeholder(tf.float32, shape=[], name="Dropout")


    def add_word_embeddings_op(self):
        '''
        Add word embedings to model
        '''
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32, trainable=False)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            # get embeddings matrix
            _char_embeddings = tf.Variable(tf.random_uniform([self.nchars, self.cfg.CHAR_EMB_DIM], -1.0, 1.0),
                                           name="_char_embeddings",
                                           dtype=tf.float32)
            self.char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                     self.char_ids,
                                                     name="char_embeddings")
            s = tf.shape(self.char_embeddings)
            self.char_embeddings = tf.reshape(self.char_embeddings, [-1, self.cfg.MAX_LENGTH_WORD, self.cfg.CHAR_EMB_DIM])
            self.embedded_chars_expanded = tf.expand_dims(self.char_embeddings, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(self.cfg.FILTER_SIZE):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.cfg.CHAR_EMB_DIM, 1, self.cfg.N_FILTERS]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_char")
                    b = tf.Variable(tf.constant(0.1, shape=[self.cfg.N_FILTERS]), name="b_char")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.cfg.MAX_LENGTH_WORD - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.cfg.N_FILTERS * len(self.cfg.FILTER_SIZE)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, s[1], num_filters_total])
            word_embeddings = tf.concat([word_embeddings, self.h_pool_flat], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """
        add pad to the data and build feed data for tensorflow
        :param words: data
        :param labels: labels
        :param lr: learning rate
        :param dropout: dropout probability
        :return: padded data with their corresponding length
        """
        char_ids, word_ids = zip(*words)
        word_ids, sentences_lengths = pad_sequences(word_ids, 0, type='sentences')
        char_ids, word_lengths = pad_sequences(char_ids, pad_token=0, type='words')

        feed = {
            self.word_ids: word_ids,
            self.sentences_lengths: sentences_lengths,
            self.char_ids: char_ids,
            self.word_lengths: word_lengths
        }

        if labels is not None:
            labels, _ = pad_sequences(labels, 0, type='sentences')
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sentences_lengths


    def add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.cfg.HIDDEN_SIZE)
            cell_bw = tf.contrib.rnn.LSTMCell(self.cfg.HIDDEN_SIZE)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw, self.word_embeddings,
                                                                        sequence_length=self.sentences_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2 * self.cfg.HIDDEN_SIZE, self.ntags],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.cfg.HIDDEN_SIZE])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])


    def add_loss_op(self):
        """
        Adds loss to self
        """
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.sentences_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)


    def add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)


    def predict_batch(self, sess, words, labels):
        """
        Args:
            sess: a tensorflow session
            words: list of sentences
        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        """
        # get the feed dictionnary
        fd, sequence_lengths = self.get_feed_dict(words, labels, dropout=1.0)

        labels_pred, loss = sess.run([self.labels_pred, self.loss], feed_dict=fd)

        return labels_pred, sequence_lengths, loss


    def run_evaluate(self, sess, test, tags):
        """
        Evaluates performance on test set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
            tags: {tag: index} dictionary
        Returns:
            accuracy
            f1 score
        """
        accs = []
        losses = 0.0
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in batch_gen(test, self.cfg.BATCH_SIZE):
            labels_pred, sequence_lengths, loss = self.predict_batch(sess, words, labels)
            losses += loss
            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length] #TODO: it is useless!
                lab_pred = lab_pred[:length]
                accs += [a==b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, tags))
                lab_pred_chunks = set(get_chunks(lab_pred, tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1, losses
