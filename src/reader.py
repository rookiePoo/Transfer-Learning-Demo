#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import Sequence, to_categorical
from config import config as cfg
#自己定义的图像增强，也可以使用imgaug库
#可参考https://zhuanlan.zhihu.com/p/44673440
#https://imgaug.readthedocs.io/en/latest/index.html
from img_aug import ImageAugment
from PIL import Image
import os

import numpy as np

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, data_list, labels = None, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=10, augment = False, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_list = data_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data batch
        data_batch = [self.data_list[k] for k in indexes]
        #print(data_batch)
        #return data_batch
        X, y = self.__data_generation(data_batch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        #print(self.indexes)

    def __data_generation(self, data_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, img_data in enumerate(data_batch):
            # get
            imgpath, label = img_data.strip().split('\t')
            imgpath = os.path.join(cfg.img_data_dir, imgpath)
            img = Image.open(imgpath).convert('RGB')
            if self.augment:
                img = ImageAugment(img).img
                img.show()
            img = img.resize(self.dim, Image.ANTIALIAS)
            img = np.array(img, dtype="float") / 255.0

            X[i,] = img

            # class label
            y[i] = int(label)

        return X, to_categorical(y, num_classes=self.n_classes)


if  __name__ == "__main__":
    trainfp = open(cfg.train_file_path, 'r')
    train_list = trainfp.readlines()[:11]
    epoches = 3
    for e in range(epoches):
        dg = DataGenerator(train_list, batch_size=4,augment = True)
        print('epoch: ', e)
        steps = dg.__len__()
        for s in range(steps):
            dg.__getitem__(s)
