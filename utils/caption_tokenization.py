from keras import preprocessing
from keras.preprocessing import text

from utils import constants


class CaptionTokenizer:
    def __init__(self, top_keys=constants.TOP_KEYS):
        self.tokenizer = text.Tokenizer(num_words=top_keys, oov_token=constants.UNKNOWN, filters=constants.FILTERS)

    @staticmethod
    def __max_lenght(captions):
        return max(len(c) for c in captions)

    def __fit_transform(self, captions):
        self.tokenizer.fit_on_texts(captions)
        train_seqs = self.tokenizer.texts_to_sequences(captions)
        return train_seqs

    def __pad_sequences(self, train_seqs):
        self.tokenizer.word_index[constants.PAD] = 0
        self.tokenizer.index_word[0] = constants.PAD
        captions_vector = preprocessing.sequence.pad_sequences(train_seqs, padding='post')
        return captions_vector

    def tokenize(self, captions):
        train_seqs = self.__fit_transform(captions=captions)
        captions_vector = self.__pad_sequences(train_seqs=train_seqs)
        max_lenght = self.__max_lenght(captions=captions_vector)
        return max_lenght, captions_vector




