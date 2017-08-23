import numpy as np
import os
from config import DefaultConfig as cfg

class conllReader(object):
    """
    This class will iterate over CoNLL dataset.
    """

    def __init__(self, filename, processing_word=None, processing_tag=None):

        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.length = None

    def __iter__(self):
        with open(self.filename, encoding='utf-8') as f:
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
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
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
    vocab_words = dict()
    vocab_tags = dict()
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
    vocab_char = dict()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_sentences(datasets):
    """
    :param dataset: an iterator yielding tuples (sentence, tags)
    :return: a list of sentences
    """
    sentences = []
    for dataset in datasets:
        for sentence, _ in dataset:
            sentences.append(sentence)

    return sentences

def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))

def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        print("Error loading file")
    return d

def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = cfg.NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[cfg.UNK]

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f
