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

# 将验证码图片转换为tfrecords格式

import config
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.platform import gfile
import warnings

warnings.filterwarnings("ignore")
image_height = config.IMAGE_HEIGHT
image_width = config.IMAGE_WIDTH
char_sets = config.CHAR_SETS
classes_num = config.CLASSES_NUM
chars_num = config.CHARS_NUM
record_dir = config.RECORD_DIR
train_file = config.TRAIN_FILE
valid_file = config.VALID_FILE
test_path = config.TEST_PATH
train_path = config.TRAIN_PATH
valid_path = config.VALID_PATH


# 将数字转化为int64类型的对象
def _int64_feature(value):
    return tf.train.Feature(int64_list=
                            tf.train.Int64List(value=[value]))


# 将数字转化为bytes类型对象
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=
                            tf.train.BytesList(value=[value]))


# 将标签转化为特征向量
def label_to_one_hot(label):
    # 定义一个用0填充的数组,大小为:(验证码位数,验证码字符类型个数)
    one_hot_label = np.zeros([chars_num, classes_num])
    offset = []
    index = []
    for i, j in enumerate(label):
        offset.append(i)
        index.append(char_sets.index(j))
    one_hot_index = [offset, index]
    one_hot_label[one_hot_index] = 1.0
    return one_hot_label.astype(np.uint8)


# 将照片转化为tfrecords格式
def conver_to_tfrecords(data_set, name):
    # 如果该文件夹找不到，则创建该文件夹
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    filename = os.path.join(record_dir, name)
    print('>> 正在写入...', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    num_examples = len(data_set)
    # print(num_examples)
    for index in range(num_examples):
        image = data_set[index][0]
        # 获取图片的高和宽
        height, width = image.shape[:2]
        # 将图片转换为字符串描述
        image_raw = image.tostring()
        label = data_set[index][1]
        label_raw = label_to_one_hot(label).tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'label_raw': _bytes_feature(label_raw),
                'image_raw': _bytes_feature(image_raw)
            }))
        # print(example)
        writer.write(example.SerializeToString(0))
    writer.close()
    print('>> 写入成功!')


# 给照片打标签
def create_data_list(image_dir):
    if not gfile.Exists(image_dir):
        print(f'{image_dir} 路径不存在')
        return None
    extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
    print(f'>> 正在在 {image_dir} 中寻找图片...')
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        # print(file_glob)
        '''
        tf.gfile.Glob(file_glob)的用法及作用
        查找匹配pattern的文件并以列表的形式返回，file_glob可
        以是一个具体的文件名，也可以是包含通配符的正则表达式
        '''
        file_list.extend(gfile.Glob(file_glob))
    if not file_list:
        print(f'{image_dir} 中没有匹配到照片!')
        return None
    # 定义空列表用来存放照片的转化的数组
    images = []
    # 定义空列表用来存放存放照片的标签
    labels = []
    # 将file_list中的照片遍历出来
    for file_name in file_list:
        # 每张图片的详细信息
        image = Image.open(file_name)
        # 将RGB格式的图片转化为灰度图像
        image_gray = image.convert('L')
        # 将图像大小缩放为128*48
        image_resize = image_gray.resize(
            size=(image_width, image_height))
        # 将图像转化为int16类型的数组
        input_image = np.array(image_resize, dtype='int16')
        image.close()
        # 给每一张照片打标签
        label_name = os.path.basename(file_name).split('_')[0]
        images.append(input_image)
        labels.append(label_name)
        '''
        a = [1,2,3,4]
        b = ['a','b','c','d']
        list(zip(a,b))   # [(1,'a'),(2,'b'),(3,'c'),(3,'d')]
        '''
    return list(zip(images, labels))


# 主函数
def main():
    # 测试集
    training_data = create_data_list(train_path)
    # print(training_data)
    conver_to_tfrecords(training_data, train_file)

    # 验证集
    validation_data = create_data_list(valid_path)
    conver_to_tfrecords(validation_data, valid_file)

if __name__ == '__main__':
    main()
