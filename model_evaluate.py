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

# 评估模型

from datetime import datetime
import sys
import math
import config
import tensorflow as tf
import captcha_model as model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 100
num_examples = 20000
checkpoint_dir = config.CHECKPOINT_DIR


def run_eval():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        images, labels = model.inputs(train=False, batch_size=batch_size)
        logits = model.inference(images, keep_prob=1)
        eval_correct = model.evaluation(logits, labels)
        # 创建会话
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            num_iter = int(math.ceil(num_examples / batch_size))
            true_count = 0
            total_true_count = 0
            total_sample_count = num_iter * batch_size
            step = 0
            print('>> 循环次数: %d, 总个数 : %d' % (num_iter, total_sample_count))
            while step < num_iter and not coord.should_stop():
                true_count = sess.run(eval_correct)
                total_true_count += true_count
                precision = true_count / batch_size
                print('>> 时间: %s, 第 %d 步: 正确/总数: %d/%d 精度为 @ 1 = %.3f'
                      % (datetime.now(), step, true_count, batch_size, precision))
                step += 1
            precision = total_true_count / total_sample_count
            print('>> 时间: %s, 正确/总数: %d/%d, 精度为 @ 1 = %.3f'
                  % (datetime.now(), total_true_count, total_sample_count, precision))
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    run_eval()
