#!/usr/bin/env python3
# coding=utf-8
'''
name : ZhouLiang
email : Brookzhoul@163.com
data : 2019-2-14
company :http://www.dltxsoft.com/
project : captcha recognize
env : python3.6
'''

# 验证码识别

import os
import sys
import config
import numpy as np
import tensorflow as tf
import captcha_model as model
from datetime import datetime
from PIL import Image
from tensorflow.python.platform import gfile
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
image_width = config.IMAGE_WIDTH
image_height = config.IMAGE_HEIGHT
char_sets = config.CHAR_SETS
classes_num = config.CLASSES_NUM
chars_num = config.CHARS_NUM
checkpoint_dir = config.CHECKPOINT_DIR
captcha_dir = config.TEST_PATH

# 将one-hot编码变为文本
def one_hot_to_tests(recog_result):
    texts = []
    for i in range(recog_result.shape[0]):
        index = recog_result[i]
        texts.append(''.join([char_sets[i] for i in index]))
    return texts

def input_data(image_dir):
    if not gfile.Exists(image_dir):
        print(f'>> {image_dir} 文件夹没有图片')
        return None
    extensions = ['png', 'tif']
    print(f'>> 正在在 {image_dir} 文件夹寻找图片...')
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(gfile.Glob(file_glob))
    if not file_list:
        print(f'在 {image_dir} 文件夹没有找到文件!')
        return None
    batch_size = len(file_list)
    images = np.zeros([batch_size, image_height*image_width], dtype='float32')
    files = []
    i = 0
    for file_name in file_list:
        # 打开图片
        image = Image.open(file_name)
        # 图像灰度化
        image = image.convert('L')
        # 图像二值化
        # threshold = 230
        # table = []
        # for i in range(256):
        #     if i<threshold:
        #         table.append(0)
        #     else:
        #         table.append(1)
        # image = image.point(table, '1')
        image_resize = image.resize(size=(image_width, image_height))
        image.close()
        input_img = np.array(image_resize,dtype='float32')
        input_img = np.multiply(input_img.flatten(), 1./255) - 0.5
        images[i,:] = input_img
        base_name = os.path.basename(file_name)
        files.append(base_name)
        i += 1
    return images, files


def run_predict():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_images, input_filenames = input_data(captcha_dir)
        images = tf.constant(input_images)
        logits = model.inference(images, keep_prob=1)
        result = model.output(logits)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        print(tf.train.latest_checkpoint(checkpoint_dir))
        recog_result = sess.run(result)
        sess.close()
        text = one_hot_to_tests(recog_result)
        total_count = len(input_filenames)
        true_count = 1000
        for i in range(total_count):
            print(f'{input_filenames[i]} 的识别结果为-----> {text[i]}')
            if text[i] in input_filenames[i]:
                true_count -= 1
        precision = true_count / total_count
        print('时间: %s, 正确个数/总数: %d/%d, 识别率: %.3f' %
              (datetime.now(), true_count, total_count, precision))

# 主函数
def main():
    run_predict()

if __name__ == '__main__':
    main()














