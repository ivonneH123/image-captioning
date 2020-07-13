from keras import Model
from keras.applications import InceptionV3

from config import Environment

from utils import DataLoader, ImageLoader, constants


def train(base_image_mode=InceptionV3):
    # Load environment variables
    e = Environment()

    # Start time

    # initialize loaders
    data_loader = DataLoader(data_type=e.data_type, annotation_file=e.captions_file, data_dir=e.data_dir, limit=10000)
    image_loader = ImageLoader()

    # Loading and preprocessing data
    captions, image_paths = data_loader.load()

    # Initialize feature extraction model
    image_model = base_image_mode(include_top=False, weights=constants.IMAGE_WEIGHTS)
    image_model = Model(image_model.input, image_model.layers[-2].output)

    for image_path in image_paths:
        img = image_loader(image_path)


if __name__ == "__main__":
    train()
