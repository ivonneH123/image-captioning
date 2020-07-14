import logging
import os
import sys
from pickle import Unpickler

from model.keras import ImageCaptioningModel
from utils import data_generator, constants

log = logging.getLogger(__name__)


class Coach:
    def __init__(self,max_length, vocabulary_size, model=ImageCaptioningModel, generator=data_generator):
        self.generator = generator
        self.max_length = max_length
        self.vocabulary_size = vocabulary_size
        self.model = model(max_length=max_length, vocabulary_size=vocabulary_size)

    @staticmethod
    def __load_features_dict(image_dict_path):
        if not os.path.isfile(image_dict_path):
            log.error(f"File {image_dict_path} not found.")
            sys.exit()
        else:
            with open(image_dict_path, "rb") as f:
                features_dict = Unpickler(f).load()

            return features_dict

    def train(self, image_dict_path, captions_vector, epochs, batch_size, verbose):
        features_dict = self.__load_features_dict(image_dict_path)
        return self.model.fit(generator=self.generator(captions_vector, features_dict,
                                                       batch_size, self.vocabulary_size, self.max_length),
                              epochs=epochs,
                              train_size=len(captions_vector),
                              batch_size=batch_size,
                              verbose=verbose)

    def save_model(self, save_dir=constants.SAVE_DIR, filename=constants.FILENAME):
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.model.save_weights(filepath)
        return filepath



