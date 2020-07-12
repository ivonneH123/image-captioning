from pycocotools.coco import COCO

from config import Environment


e = Environment()
coco = COCO(e.captions_file)
