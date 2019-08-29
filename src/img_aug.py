#coding:utf-8

import math
from PIL import Image, ImageFilter, ImageEnhance
import random

class ImageAugment:
    """
    :param PIL.Image
    :return PIL.Image
    """
    def __init__(self, img):
        self.img = img
        self.aug_group()

    # 随机抠图, 需要修改
    def random_crop(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        bound = min((float(self.img.size[0]) / self.img.size[1]) / (w**2),
                    (float(self.img.size[1]) / self.img.size[0]) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = self.img.size[0] * self.img.size[1] * random.uniform(scale_min,
                                                                 scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, self.img.size[0] - w + 1)
        j = random.randint(0, self.img.size[1] - h + 1)

        self.img = self.img.crop((i, j, i + w, j + h))

    # 水平翻转
    def hor(self):
        self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)

    # 色彩抖动
    def jittering(self):
        i_randint = random.randint(1, 9)
        if i_randint == 1:
            # 高斯模糊
            self.img = self.img.filter(ImageFilter.GaussianBlur)
        elif i_randint == 2:
            # 普通模糊
            self.img = self.img.filter(ImageFilter.BLUR)
        elif i_randint == 3:
            # 边缘增强
            self.img = self.img.filter(ImageFilter.EDGE_ENHANCE)
        elif i_randint == 4:
            # 找到边缘
            self.img = self.img.filter(ImageFilter.FIND_EDGES)
        elif i_randint == 5:
            # 浮雕
            self.img = self.img.filter(ImageFilter.EMBOSS)
        elif i_randint == 6:
            # 轮廓
            self.img = self.img.filter(ImageFilter.CONTOUR)
        elif i_randint == 7:
            # 锐化
            self.img = self.img.filter(ImageFilter.SHARPEN)
        elif i_randint == 8:
            # 平滑
            self.img = self.img.filter(ImageFilter.SMOOTH)
        else:
            # 细节
            self.img = self.img.filter(ImageFilter.DETAIL)

    def random_brightness(self, lower=0.6, upper=1.4):
        e = random.uniform(lower, upper)
        self.img = ImageEnhance.Brightness(self.img).enhance(e)

    def random_contrast(self, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        self.img = ImageEnhance.Contrast(self.img).enhance(e)

    def random_color(self, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        self.img = ImageEnhance.Color(self.img).enhance(e)

    def rotate_image(self):
        angle = random.randint(-45, 45)
        self.img = self.img.rotate(angle)

    def aug_group(self):
        func_group = [self.random_crop, self.hor, self.jittering, self.rotate_image, \
         self.random_brightness, self.random_color, self.random_contrast]
        group_num = random.randint(0, 7)
        if group_num == 0:
            pass
        else:
            random.shuffle(func_group)
            for i in range(group_num):
                func_group[i]()

