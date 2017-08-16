import numpy as np
import os

class conllReader(object):
    """
    This class will iterate over CoNLL dataset.
    """

    def __init__(self, filename, convert_digits=True):

        self.filename = filename
        self.convert_digits = convert_digits
        self.length = None

    def __iter__(self):
        with open(self.filename,encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(words) != 0:
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split('\t')
                    word, tag = ls[1], ls[5]
                    if self.convert_digits:
                        if word.isdigit():
                            word = "$NUM$"
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_word2vec(datasets):
    """
    :param dataset: an iterator yielding tuples (sentence, tags)
    :return: a list of sentences
    """
    sentences = []
    for dataset in datasets:
        for words, _ in dataset:
            sentences.append(words)

    return sentences

