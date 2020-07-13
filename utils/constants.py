# Random state
RANDOM_STATE = 0

# Captions loading
START_SEQ = '<start>'
END_SEQ = '<end>'

# Image loading and preprocessing
IMAGE_PATH = '{images_dir}/images/COCO_{data_type}_{image_id}_012d.jpg'
INCEPTION_DIM = (299, 299)
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
