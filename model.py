import tensorflow as tf
from data_helper import pad_sequences, batch_gen, get_chunks
import numpy as np


class Model(object):
    def __init__(self, config, embeddings, ntags, nchars):
        '''
        Tensorflow model
        :param embeddings: word2vec embedding file which loaded
        :param ntags: number of tags
        :param nchars: number of chars
        '''
        self.cfg = config
        self.embeddings = embeddings
        self.nchars = nchars
        self.ntags = ntags

        self.add_placeholders()                 # Initial placeholders
        self.add_word_embeddings_op()           # add embedding operation to graph
        self.add_logits_op()                    # add logits operation to graph
        self.add_loss_op()                      # add loss operation to graph
        self.add_train_op()                     # add train (optimzier) operation to graph

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

    def add_placeholders(self):
        '''
        Initial placeholders
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
        Add word embedings + Char CNN operation to graph
        '''
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32, trainable=False)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            xavi = tf.contrib.layers.xavier_initializer
            # Get char level embeddings matrix
            _char_embeddings = tf.get_variable("_char_embeddings", shape=[self.nchars, self.cfg.CHAR_EMB_DIM],
                                               dtype=tf.float32,
                                               initializer=xavi())
            self.char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                     self.char_ids,
                                                     name="char_embeddings")
            # get shape of char embd matrix
            s = tf.shape(self.char_embeddings)
            # Reshape char_embd matrix to [batches * sentence length , max_word_length , char_embedding_size]
            self.char_embeddings = tf.reshape(self.char_embeddings, [-1, self.cfg.MAX_LENGTH_WORD, self.cfg.CHAR_EMB_DIM])
            # Add one dimension at the end of char_emb matrix to have shape like:
            # [batches, height, width, channels] like an image. Here channel=1
            self.embedded_chars_expanded = tf.expand_dims(self.char_embeddings, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            # Here we do convolution over [words x char_emb] with different filter sizes [2,3,4,5].
            for i, filter_size in enumerate(self.cfg.FILTER_SIZE):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Define convolution filter Layer with shape = [filter_height, filter_width, in_channels, out_channels]
                    filter_shape = [filter_size, self.cfg.CHAR_EMB_DIM, 1, self.cfg.N_FILTERS]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_char")
                    b = tf.Variable(tf.constant(0.1, shape=[self.cfg.N_FILTERS]), name="b_char")
                    # conv return shape= [batch * sentence_length, MAX_LENGTH_WORD - FILTER_SIZE + 1, 1, N_FILTERS]
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") # h has same shape as conv
                    # Maxpooling over the outputs
                    # return shape= [Batch_size, 1, 1, N_FILTERS]
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.cfg.MAX_LENGTH_WORD - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    # Add all convolution outputs to a list
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.cfg.N_FILTERS * len(self.cfg.FILTER_SIZE)
            # Concatinate all pooled features over 3th dimension
            self.h_pool = tf.concat(pooled_outputs, 3) # has shape= [Batch_size, 1, 1, len(FILTER_SIZE) * N_FILTERS]
            # Reshape data to shape= [Batch_Size, Sentence_Length, len(FILTER_SIZE) * N_FILTERS]
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, s[1], num_filters_total])
            # Add char features to embedding words. Shape= [Batch_Size, Sentence_Length, Word_Emb_length + len(FILTER_SIZE) * N_FILTERS]
            word_embeddings = tf.concat([word_embeddings, self.h_pool_flat], axis=-1)
        # add Dropout regularization
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
        # Unzip data to char_ids and word_ids
        char_ids, word_ids = zip(*words)
        # pad sentence to maximum sentence length of current batch
        word_ids, sentences_lengths = pad_sequences(word_ids, 0, type='sentences')
        # pad words to maximum word length of current batch
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
        Adds logits to Model. We use BiLSTM + fully connected layer to predict word sequences labels
        """
        with tf.variable_scope("bi-lstm"):
            # Define Forwards cell
            cell_fw = tf.contrib.rnn.LSTMCell(self.cfg.HIDDEN_SIZE)
            # Define Backwards cell
            cell_bw = tf.contrib.rnn.LSTMCell(self.cfg.HIDDEN_SIZE)
            # Run BiLSTM
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw, self.word_embeddings,
                                                                        sequence_length=self.sentences_lengths,
                                                                        dtype=tf.float32)
            # Concatenate Forward and backward over last axis
            # The shape is: [Batch_size, Sentence_length, 2*HIDDEN_SIZE]
            rnn_output = tf.concat([output_fw, output_bw], axis=-1)
            # Apply Dropout regularization
            rnn_output = tf.nn.dropout(rnn_output, self.dropout)

        with tf.variable_scope("proj"):
            # Define weights and Biases
            W1 = tf.get_variable("W1", shape=[2 * self.cfg.HIDDEN_SIZE, self.cfg.HIDDEN_SIZE],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

            b1 = tf.get_variable("b1", shape=[self.cfg.HIDDEN_SIZE], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            W2 = tf.get_variable("W2", shape=[self.cfg.HIDDEN_SIZE, self.ntags],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

            b2 = tf.get_variable("b2", shape=[self.ntags], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            # get sentence length
            ntime_steps = tf.shape(rnn_output)[1]
            # Reshape to 2D to calculate W1. shape= [Batch_size * sentences_length, 2*HIDDEN_SIZE]
            rnn_output = tf.reshape(rnn_output, [-1, 2 * self.cfg.HIDDEN_SIZE])
            # Apply projection, return [Batch_size * sentences_length, HIDDEN_SIZE]
            w1_output = tf.matmul(rnn_output, W1) + b1
            # Apply nonlinearity
            w1_output = tf.nn.relu(w1_output, name="w1_relu")
            # Apply Dropout regularization
            w1_output = tf.nn.dropout(w1_output, self.dropout)
            # Apply projection, return shape= [Batch_size * sentences_length, N_Tags]
            pred = tf.matmul(w1_output, W2) + b2
            # Return back to shape= [[Batch_size , sentences_length, N_Tags]
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])


    def add_loss_op(self):
        """
        Adds loss to Model
        """
        # Get highest probabilty of predicted labels
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        # Compute loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        # Use Mask to eliminate Zeros paddings
        mask = tf.sequence_mask(self.sentences_lengths)
        losses = tf.boolean_mask(losses, mask)
        # assign loss to self
        self.loss = tf.reduce_mean(losses)

        # Create a summary to monitor loss
        tf.summary.scalar("loss", self.loss)


    def add_train_op(self):
        """
        Add train_op to Model
        """
        with tf.variable_scope("train_step"):
            # In each epoch iteration, the Learning Rate will decay which defined in config file
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)


    def predict_batch(self, sess, words, labels):
        """
        Args:
            sess: a tensorflow session
            words: list of sentences
            labels: list of true labels
        Returns:
            labels_pred: list of labels for each sentence
            sequence_length: length of sentences
            loss: loss of current batch
        """
        # get the feed dictionnary
        fd, sequence_lengths = self.get_feed_dict(words, labels, dropout=1.0)
        # Run Tensorflow graph
        labels_pred, loss = sess.run([self.labels_pred, self.loss], feed_dict=fd)
        return labels_pred, sequence_lengths, loss


    def run_evaluate(self, sess, test, tags):
        """
        Evaluates performance on dev set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
            tags: {tag: index} dictionary
        Returns:
            accuracy
            f1 score
            loss
            Precision
            Recall
        This code honored to:
        https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
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
         # Create a summary to monitor accuracy
        tf.summary.scalar("accuracy", acc)
        # Create a summary to monitor Precision
        tf.summary.scalar("accuracy", p)
        # Create a summary to monitor Recall
        tf.summary.scalar("accuracy", r)
        return acc, f1, losses, p,r
