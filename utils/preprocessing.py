import numpy as np
from keras.applications import inception_v3
from keras.preprocessing import image

from utils import constants


class ImageLoader:
    def __init__(self, dim=constants.INCEPTION_DIM, model=inception_v3):
        self.dim = dim
        self.model = model

    def __load_img(self, image_path):
        img = image.load_img(image_path, target_size=self.dim)
        return img

    def __preprocess_input(self, img):
        img = image.img_to_array(img)
        img = self.model.preprocess_input(img)
        return img

    def __call__(self, image_path):
        raw_img = self.__load_img(image_path)
        proprocessed_img = self.__preprocess_input(raw_img)
        return proprocessed_img


