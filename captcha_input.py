import os
import tensorflow as tf
import config

# 变量的详细定义见config.py
record_dir = config.RECORD_DIR
train_file = config.TRAIN_FILE
valid_file = config.VALID_FILE
image_width = config.IMAGE_WIDTH
image_height = config.IMAGE_HEIGHT
classes_num = config.CLASSES_NUM
chars_num = config.CHARS_NUM


# 读取tfrecords格式的数据并转换它的格式
def read_and_decode(filename_queus):
    reader = tf.TFRecordReader()
    # 获取文件
    _, serialized_example = reader.read(filename_queus)
    # 取出包含image和label的feature对象
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string)
        })
    # 将原来编码为字符串类型的变量重新变回来
    image = tf.decode_raw(features['image_raw'], tf.int16)
    image.set_shape([image_height * image_width])
    # 将int16的类型转换为float32,但不改变原数据的值和形状
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # 将原图片的维度转变为（48,128,1)
    reshape_image = tf.reshape(image, [image_height, image_width, 1])
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label.set_shape([chars_num * classes_num])
    reshape_label = tf.reshape(label, [chars_num, classes_num])
    return tf.cast(reshape_image, tf.float32), \
           tf.cast(reshape_label, tf.float32)


def inputs(train, batch_size):
    # 获取tfrecords格式数据的路径
    filename = os.path.join(record_dir,
                            train_file if train else valid_file)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename])
        # print(filename_queue)
        image, label = read_and_decode(filename_queue)
        if train:
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=6,
                capacity=2000 + 3 * batch_size,
                min_after_dequeue=2000)
        else:
            images, sparse_labels = tf.train.batch(
                [image,label],
                batch_size=batch_size,
                num_threads=6,
                capacity=2000+3*batch_size)
        # print(images,sparse_labels)
        return images, sparse_labels



# # 用于测试以上函数
# inputs(train=True,batch_size=128)
