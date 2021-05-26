from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import json

from flask import Flask, request

import argparse
import sys
import os.path
from datetime import datetime
from PIL import Image
import numpy as np
import os
import io

import tensorflow as tf
from tensorflow.python.platform import gfile
import captcha_model as captcha

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT

CHAR_SETS = config.CHAR_SETS
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM

FLAGS = None


def one_hot_to_texts(recog_result):
    texts = []
    for i in xrange(recog_result.shape[0]):
        index = recog_result[i]
        texts.append(''.join([CHAR_SETS[i] for i in index]))
    return texts


def input_data(image_base64):
    images = np.zeros([1, IMAGE_HEIGHT*IMAGE_WIDTH], dtype='float32')
    files = []
    file_name = "cap"
    image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    image_gray = image.convert('L')
    image_resize = image_gray.resize(size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    image.close()
    input_img = np.array(image_resize, dtype='float32')
    input_img = np.multiply(input_img.flatten(), 1./255) - 0.5
    images[0, :] = input_img
    base_name = os.path.basename(file_name)
    files.append(base_name)
    return images, files


def run_predict(image_base64):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_images, input_filenames = input_data(image_base64)
        images = tf.constant(input_images)
        logits = captcha.inference(images, keep_prob=1)
        result = captcha.output(logits)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        # print(tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        recog_result = sess.run(result)
        sess.close()
        text = one_hot_to_texts(recog_result)
        total_count = len(input_filenames)
        result = ""
        for i in range(total_count):
            result = text[i]
        return result


api = Flask("hello")


@api.route('/ping', methods=['POST'])
def ping():
    data = json.loads(request.get_data(as_text=True))
    res = run_predict(data['image'])
    return res


def main(_):
    api.run(host="0.0.0.0")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./captcha_train',
        help='Directory where to restore checkpoint.'
    )
    parser.add_argument(
        '--captcha_dir',
        type=str,
        default='./data/test_data/test.jpg',
        help='Directory where to get captcha images.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
