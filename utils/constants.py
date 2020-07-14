# Random state
RANDOM_STATE = 0

# Captions loading
START_SEQ = '<start>'
END_SEQ = '<end>'

# Image loading and preprocessing
IMAGE_PATH = '{images_dir}/image/COCO_{data_type}_{image_id}.jpg'
ID_SIZE = 12
INCEPTION_DIM = (299, 299)
INCEPTION_OUTPUT = 2048
LIMIT = 10000

# Image model
IMAGE_WEIGHTS = 'imagenet'
TEMP_DIR = 'temp/'
FEATURE_DICT_NAME = 'feature_dict'

# Tokenization
TOP_KEYS = 6000
FILTERS = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
UNKNOWN = '<unknown>'
PAD = '<pad>'


# Captioning preprocessing
TEST_SIZE = 0.25

# Image captioning model
DROPOUT = 0.5
EMBEDDING_DIM = 256
LEARNING_RATE = 0.01
EPOCHS = 30
BATCH_SIZE = 32
VERBOSE = True
SAVE_DIR = 'model/weights'
FILENAME = 'model_{epochs}epochs_{name}'
