# about captcha image
IMAGE_HEIGHT = 25
IMAGE_WIDTH = 80
CHAR_SETS = '0123456789'
CLASSES_NUM = len(CHAR_SETS)
CHARS_NUM = 4
# for train
RECORD_DIR = './data'
TRAIN_FILE = 'train.tfrecords'
VALID_FILE = 'valid.tfrecords'