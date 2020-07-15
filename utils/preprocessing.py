import numpy as np
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image

from utils import constants


class ImageLoader:
    def __init__(self, dim=constants.INCEPTION_DIM, preprocess_input=inception_v3.preprocess_input):
        self.dim = dim
        self.preprocess_input = preprocess_input

    def _load_img(self, image_path):
        img = image.load_img(image_path, target_size=self.dim)
        return img

    def _preprocess_input(self, img):
        img = image.img_to_array(img)
        img = self.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img

    def load(self, image_path):
        raw_img = self._load_img(image_path)
        proprocessed_img = self._preprocess_input(raw_img)
        return proprocessed_img


