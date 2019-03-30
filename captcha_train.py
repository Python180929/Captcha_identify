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

#模型的训练

import time
import sys
import config
import tensorflow as tf
import captcha_model as model
from datetime import datetime
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

batch_size = config.BATCH_SIZE
checkpoint = config.CHECKPOINT
train_dir = config.TRAIN_DIR


# 对验证码模型进行训练
def run_train():
    with tf.Graph().as_default():
        images, labels = model.inputs(
            train=True, batch_size=batch_size)
        # 返回5*62的向量，5代表识别结果的位数，62是每一位上可能的结果（数字，大小写字母）
        logits = model.inference(images, keep_prob=0.5)
        loss = model.loss(logits, labels)
        train_op = model.training(loss)
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        # 创建会话
        sess = tf.Session()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        try:
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                if step % 10 == 0:
                    print('>> 第%d次训练, 损失 = %.2f, 所用时间 = %0.3f秒' %
                          (step, loss_value, duration))
                if step % 100 == 0:
                    print(f'>> {datetime.now()} 文本保存在 {checkpoint}')
                    saver.save(sess, checkpoint, global_step=step)
                step += 1
        except Exception as e:
            print(f'>> {datetime.now()} 文本保存在 {checkpoint}')
            # 向文件夹中写入包含当前模型中所有可训练变量的checkpoint文件
            saver.save(sess, checkpoint, global_step=step)
            # 线程停止
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
        # 关闭会话
        sess.close()


# 主函数
def main():
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    run_train()


if __name__ == '__main__':
    main()

