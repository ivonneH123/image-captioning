from keras import Model
from keras.applications import InceptionV3

from config import Environment

from utils import DataLoader
from utils.feature_extraction import FeatureExtractor


def train(base_image_mode=InceptionV3):
    # Load environment variables
    e = Environment()

    # Start time

    # initialize data loader
    data_loader = DataLoader(data_type=e.data_type, annotation_file=e.captions_file, data_dir=e.data_dir, limit=10000)

    # Loading and preprocessing data
    captions, image_paths = data_loader.load()

    # Extract image features
    extractor = FeatureExtractor(image_paths=image_paths, base_image_mode=base_image_mode)
    extractor()


if __name__ == "__main__":
    train()
