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

import numpy as np
import tensorflow as tf
from PIL import Image
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class MalaiCaprcha:
    """
    马来西亚验证码的识别
        6位数字、英文，区分大小写
    """

    def __init__(self):
        self.image_width = 150
        self.image_height = 40
        self.char_sets = 'abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIGJKLMNOPQRSTUVWXYZ'
        self.classes_num = len(self.char_sets)
        self.chars_num = 6
        # 模型存放地址
        self.model_path = './captcha_model'

    # 构建卷积层
    def conv2d(self, value, weight):
        '''该函数用于返回一个二维卷积层'''
        # tf.nn.conv2d() 函数实现卷积操作
        # padding='SAME'会对图像边缘补0,完成图像上所有像素（特别是边缘象素）的卷积操作
        return tf.nn.conv2d(value, weight,
                            strides=[1, 1, 1, 1], padding='SAME')

    # 构建池化层
    def max_pool_2x2(self, value, name):
        '''max_pool_2x2将特征映射向下采样2倍'''
        # tf.nn.max_pool()函数实现最大池化操作，进一步提取图像的抽象特征，并且降低特征维度
        # ksize=[1, 2, 2, 1]定义最大池化操作的核尺寸为2*2
        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)

    def weight_variable(self, name, shape):
        '''weight_variable生成给定变量的权值变量'''
        with tf.device('/cpu:0'):
            initializer = tf.truncated_normal_initializer(stddev=0.1)
            # print(initializer)
            var = tf.get_variable(name, shape, initializer=initializer,
                                  dtype=tf.float32)
        return var

    def bias_variable(self, name, shape):
        '''bias_variable生成给定形状的偏差变量'''
        with tf.device('cpu:0'):
            initializer = tf.constant_initializer(0.1)
            var = tf.get_variable(name, shape, initializer=initializer,
                                  dtype=tf.float32)
        return var

    def inference(self, images, keep_prob):
        # 获得输入
        images = tf.reshape(images, [-1, self.image_height, self.image_width, 1])
        # 卷积层1
        with tf.variable_scope('conv1') as scope:
            # shape[3, 3, 1, 32]
            # 里前两个参数表示卷积核尺寸大小，即patch;
            # 第三个参数是图像通道数;
            # 第四个参数是该层卷积核的数量，有多少个卷积核就会输出多少个卷积特征图像
            kernel = self.weight_variable('weights', shape=[3, 3, 1, 64])
            # 每个卷积核都配置一个偏置量，该层有多少个输出，就应该配置多少个偏置量
            biases = self.bias_variable('biases', [64])
            # tf.nn.bias_add() 函数的作用是将偏置项biases加到卷积结果上去;
            # 注意这里的偏置项biases必须是一维的，并且数量一定要与卷积结果最后一维数量相同
            pre_activation = tf.nn.bias_add(self.conv2d(images, kernel), biases)
            # tf.nn.relu() 函数是relu激活函数，实现输出结果的非线性转换;
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # 池化层1
        pool1 = self.max_pool_2x2(conv1, name='pool1')
        # 原图像height = 48, WIDTH = 128,
        # 经过神经网络第一层卷积（图像尺寸不变、特征×64）、池化（图像尺寸缩小一半，特征不变）之后;
        # 输出大小为 24*64*64
        # print(pool1)

        # 卷积层2
        with tf.variable_scope('conv2') as scope:
            kernel = self.weight_variable('weights', shape=[3, 3, 64, 64])
            biases = self.bias_variable('biases', [64])
            pre_activation = tf.nn.bias_add(self.conv2d(pool1, kernel), biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
        # 池化层2
        pool2 = self.max_pool_2x2(conv2, name='pool2')
        # 原图像HEIGHT = 60 WIDTH = 160，经过神经网络第一层后输出大小为 24*64*64
        # 经过神经网络第二层运算后输出为 12*32*64
        # 24*64的图像经过3*3的卷积核池化，padding为SAME，输出维度是12*32
        # print(pool2)

        # 卷积层3
        with tf.variable_scope('conv3') as scope:
            kernel = self.weight_variable('weights', shape=[3, 3, 64, 64])
            biases = self.bias_variable('biases', [64])
            pre_activation = tf.nn.bias_add(self.conv2d(pool2, kernel), biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
        # 池化层3
        pool3 = self.max_pool_2x2(conv3, name='pool3')
        # 原图像HEIGHT = 60 WIDTH = 160，
        # 经过神经网络第一层后输出大小为 24*64*64 ;
        # 经过神经网络第二层运算后输出为 12*32*64 ;
        # 经过神经网络第三层后输出大小为 6*16*64 ;
        # print(pool3)

        # 卷积层4
        with tf.variable_scope('conv4') as scope:
            kernel = self.weight_variable('weights', shape=[3, 3, 64, 64])
            biases = self.bias_variable('biases', [64])
            pre_activation = tf.nn.bias_add(self.conv2d(pool3, kernel), biases)
            conv4 = tf.nn.relu(pre_activation, name=scope.name)
        # 池化层4
        pool4 = self.max_pool_2x2(conv4, name='pool4')
        # 经过神经网络第一层后输出大小为 24*64*64 ;
        # 经过神经网络第二层运算后输出为 12*32*64 ;
        # 经过神经网络第三层后输出大小为 6*16*64 ;
        # 经过神经网络第四层后输出大小为 3*8*64 ;
        # print(pool4)

        # 搭建全连接层
        with tf.variable_scope('local1') as scope:
            # images.get_shape()作用是把张量images的形状转换为元组tuple的形式
            batch_size = images.get_shape()[0].value
            reshape = tf.reshape(pool4, [batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = self.weight_variable('weights', shape=[dim, 1024])
            biases = self.bias_variable('biases', [1024])
            # tf.matmul()函数是矩阵相乘
            local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                                name=scope.name)
        # tf.nn.dropout是tf里为了防止或减轻过拟合而使用的函数，一般用在全连接层;
        # Dropout机制就是在不同的训练过程中根据一定概率
        # （大小可以设置，一般情况下训练推荐0.5）随机扔掉（屏蔽）一部分神经元，
        # 不参与本次神经网络迭代的计算（优化）过程，权重保留但不做更新;
        # tf.nn.dropout()中 keep_prob用于设置概率，需要是一个占位变量，在执行的时候具体给定数值
        local1_drop = tf.nn.dropout(local1, keep_prob)

        with tf.variable_scope('softmax_linear') as scope:
            weights = self.weight_variable(
                'weights', shape=[1024, self.chars_num * self.classes_num])
            biases = self.bias_variable('biases', [self.classes_num * self.chars_num])
            softmax_linear = tf.add(tf.matmul(local1_drop, weights),
                                    biases, name=scope.name)
        # 返回6*62的向量，6代表识别结果的位数，62是每一位上可能的结果（数字，大小写字母）
        return tf.reshape(softmax_linear, [-1, self.chars_num, self.classes_num])

    # 将one-hot编码变为文本
    def one_hot_to_tests(self, recog_result):
        texts = []
        for i in range(recog_result.shape[0]):
            index = recog_result[i]
            texts.append(''.join([self.char_sets[i] for i in index]))
        return texts

    # 图片预处理
    def imageProcess(self, img):
        with open('code.png', 'wb') as f:
            f.write(img)
        image = Image.open('code.png')
        # 图像灰度化
        gray_image = image.convert('L')
        # 修改图像的大小
        image_resize = gray_image.resize(size=(self.image_width, self.image_height))
        image.close()
        # 将图片转化为numpy数组
        input_img = np.array(image_resize, dtype='float32')
        input_img = np.multiply(input_img.flatten(), 1. / 255) - 0.5
        images = np.zeros([1, self.image_height * self.image_width], dtype='float32')
        images[0, :] = input_img
        return images

    # 验证码的识别
    def codeDemo(self, img):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            input_images = self.imageProcess(img)
            images = tf.constant(input_images)
            logits = self.inference(images, keep_prob=1)
            result = tf.argmax(logits, 2)
            saver = tf.train.Saver()
            sess = tf.Session()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            recog_result = sess.run(result)
            sess.close()
            text = self.one_hot_to_tests(recog_result)
        return text[0]


if __name__ == "__main__":
    a = MalaiCaprcha()
    import requests

    url = "https://www.windowmalaysia.my/evisa/captchaImaging?_=1553594761349"
    sessions = requests.session()
    sessions.get('https://www.windowmalaysia.my/evisa/evisa.jsp?alreadyCheckLang=1&lang=zh')
    img = sessions.get(url).content
    res = a.codeDemo(img)
    print(res)
