from keras.applications import InceptionV3
from sklearn.model_selection import train_test_split

from config import Environment

from utils import DataLoader, constants
from utils.caption_tokenization import CaptionTokenizer
from utils.feature_extraction import FeatureExtractor


def train(base_image_mode=InceptionV3):
    # Load environment variables
    e = Environment()

    # Start time

    # initialize data loader
    data_loader = DataLoader(data_type=e.data_type, annotation_file=e.captions_file,
                             data_dir=e.data_dir, limit=constants.LIMIT)

    # Loading and preprocessing data
    captions, image_paths = data_loader.load()

    # Extract image features
    extractor = FeatureExtractor(image_paths=image_paths, base_image_mode=base_image_mode)
    extractor()

    # Tokenize image captions
    tokenizer = CaptionTokenizer()
    max_lenght, captions_vector = tokenizer.tokenize(captions=captions)

    # Create train and validation sets
    paths_train, paths_val, captions_train, captions_val = train_test_split(image_paths,
                                                                            captions_vector,
                                                                            test_size=constants.TEST_SIZE,
                                                                            random_state=constants.RANDOM_STATE)


if __name__ == "__main__":
    train()
