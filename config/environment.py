import os
from dotenv import load_dotenv


class Environment:
    def __init__(self):
        load_dotenv()
        self.data_dir = os.getenv('DATA_DIR')
        self.data_type = os.getenv('DATA_TYPE')
        self.ann_file = '{}/annotations/instances_{}.json'.format(self.data_dir, self.data_type)
