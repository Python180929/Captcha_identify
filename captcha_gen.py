#!/usr/bin/env python3
# coding=utf-8
'''
name : ZhouLiang
email : Brookzhoul@163.com
data : 2019-2-12
company :http://www.dltxsoft.com/
project : captcha recognize
env : python3.6
'''

# 准备验证码图片
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
import random
import config
from captcha.image import ImageCaptcha


image_height = config.IMAGE_HEIGHT
image_width = config.IMAGE_WIDTH
# 验证码长度
chars_num = config.CHARS_NUM
# 测试集数据保存路径
test_path = config.TEST_PATH
# 训练集数据保存路径
train_path = config.TRAIN_PATH
# 验证集数据保存路径
valid_path = config.VALID_PATH
# 测试集验证码个数
test_size = 1000
# 训练集验证码个数
train_size = 100000
# 验证集验证码个数
valid_size = 20000


def getRandomColor():
    '''获取一个随机色(r,g,b)格式的'''
    c1 = random.randint(160, 255)
    c2 = random.randint(0, 250)
    c3 = random.randint(5, 255)
    return (c1, c2, c3)


def getRandomStr():
    '''获取一个随机字符串，每个字符的颜色也是随机的'''
    random_num = str(random.randint(0, 9))
    random_low_alpha = chr(random.randint(97, 122))
    random_upper_alpha = chr(random.randint(65, 90))
    random_char = random.choice([random_num, random_low_alpha, random_upper_alpha])
    return random_char


def genCaptcha(gen_dir, chars_num,total_size):
    # 如果gen_dir不存在，则创建该路径
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    for i in range(total_size):
        # 获取一个Image对象，参数分别是RGB模式。宽150，高40，白色
        image = Image.new('RGB', (150, 40), (255, 255, 255))
        # 获取一个画笔对象，将图片对象传过去
        draw = ImageDraw.Draw(image)
        # 获取一个font字体对象参数是ttf的字体文件的目录，以及字体的大小
        # font = ImageFont.truetype("AngroEF-Bold.otf", size=32)
        font = ImageFont.truetype("./font/waree-bold.ttf", size=28)
        label = ''
        for j in range(chars_num):
            # 循环6次，获取6个随机字符串
            random_char = getRandomStr()
            label += random_char
            # 在图片上一次写入得到的随机字符串,参数是：定位，字符串，颜色，字体
            if j % 2 == 0:
                draw.text((j * 25, -10), random_char, getRandomColor(), font=font)
            if j % 2 != 0:
                draw.text((j * 25, 0), random_char, getRandomColor(), font=font)
        image.save(open(f"{gen_dir}/{label}_num{str(i)}.png",'wb'),'png')


if __name__ == '__main__':
    print(f'>> 正在生成{test_size}个验证码并保存在 {test_path} 文件夹中作为测试集...')
    genCaptcha(test_path, chars_num, test_size)
    print('>> 测试集验证码生成完成!')
    print(f'>> 正在生成{train_size}个验证码并保存在 {train_path} 文件夹中作为训练集...')
    genCaptcha(train_path, chars_num, train_size)
    print('>> 训练集验证码生成完成!')
    print(f'>> 正在生成{valid_size}个验证码并保存在 {valid_path} 文件夹中作为验证集...')
    genCaptcha(valid_path, chars_num, valid_size)
    print('>> 验证集验证码生成完成!')

