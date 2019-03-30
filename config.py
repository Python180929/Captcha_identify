# 验证码高度
IMAGE_HEIGHT = 40
# 验证码宽度
IMAGE_WIDTH = 150
# 验证码包含26个大小写，10个数字
CHAR_SETS = 'abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIGJKLMNOPQRSTUVWXYZ'
# 验证码包含字符的个数
CLASSES_NUM = len(CHAR_SETS)
# 验证码的位数
CHARS_NUM = 6
# 测试集数据保存路径
TEST_PATH = './data/test_data'
# 训练集数据保存路径
TRAIN_PATH = './data/train_data'
# 验证集数据保存路径
VALID_PATH = './data/valid_data'
# tfrecords格式文件存放目录
RECORD_DIR = './data'
# 训练集tfrecords格式文件名
TRAIN_FILE = 'train.tfrecords'
# 验证集tfrecords格式文件名
VALID_FILE = 'valid.tfrecords'
# 批量大小
BATCH_SIZE = 150
# 模型存放文件
CHECKPOINT = './captcha_train/captcha'
# 模型存放目录
TRAIN_DIR = './captcha_train'
CHECKPOINT_DIR = './captcha_train'