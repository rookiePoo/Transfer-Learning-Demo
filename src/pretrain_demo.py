from vgg16_keras import VGG16_FC
from config import config as cfg
from reader import DataGenerator
from keras.optimizers import Adam, SGD
from keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("Agg")

def plot_history(history):
    H = history
    plt.style.use("ggplot")
    plt.figure()
    N = cfg.epoches
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    #plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    #plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("fintune.png")


def pretrain_model(train_list, test_list, optimizer="adam", pretrained_model_path=None):
    if pretrained_model_path:
        vgg16 = load_model(pretrained_model_path)
    else:
        vgg16 = VGG16_FC(img_size=cfg.img_size, img_channels=cfg.channel_num, class_num=cfg.class_num).vgg16_fc
    print(vgg16.summary())

    train_generator = DataGenerator(train_list, dim=cfg.img_size, n_classes=cfg.class_num, batch_size=cfg.batch_size, augment=False)
    # 测试集augment=False，训练集随意
    test_generator = DataGenerator(test_list, dim=cfg.img_size, n_classes=cfg.class_num, batch_size=cfg.batch_size, augment=False)
    # 定义loss
    loss = "categorical_crossentropy"
    # 选择优化器
    if optimizer == "sgd":
        optimizer = SGD(lr=cfg.learning_rate/10, momentum=0.95)
    else:
        optimizer = Adam(lr=cfg.learning_rate, decay=cfg.learning_rate / cfg.epoches)

    vgg16.compile(loss=loss, optimizer=optimizer,
                  metrics=["accuracy"])
    # 由于自定义的生成器中已经计算了steps_per_epoch，因此这里不需要
    #history = vgg16.fit_generator(train_generator, steps_per_epoch=None, epochs=cfg.epoches, verbose=1, validation_data=test_generator)
    history = vgg16.fit_generator(test_generator, steps_per_epoch=None, epochs=cfg.epoches, verbose=1,
                                  validation_data=None)

    vgg16.save(pretrained_model_path)
    plot_history(history)

def fintune_model(train_list, test_list, fintune_model_path, fintune_save_path, optimizer="sgd"):
    vgg16 = load_model(fintune_model_path)
    vgg16 = VGG16_FC(vgg16_fc=vgg16, frozen_layers=15, img_size=cfg.img_size, img_channels=cfg.channel_num, class_num=cfg.class_num).vgg16_fc
    train_generator = DataGenerator(train_list, dim=cfg.img_size, n_classes=cfg.class_num, batch_size=cfg.batch_size,
                                    augment=False)
    # 测试集augment=False，训练集随意
    test_generator = DataGenerator(test_list, dim=cfg.img_size, n_classes=cfg.class_num, batch_size=cfg.batch_size,
                                   augment=False)
    # 定义loss
    loss = "categorical_crossentropy"
    # 选择优化器
    if optimizer == "sgd":
        optimizer = SGD(lr=cfg.learning_rate / 10, momentum=0.95)
    else:
        optimizer = Adam(lr=cfg.learning_rate, decay=cfg.learning_rate / cfg.epoches)

    vgg16.compile(loss=loss, optimizer=optimizer,
                  metrics=["accuracy"])
    # history = vgg16.fit_generator(train_generator, steps_per_epoch=None, epochs=cfg.epoches, verbose=1, validation_data=test_generator)
    history = vgg16.fit_generator(test_generator, steps_per_epoch=None, epochs=cfg.epoches, verbose=1,
                                  validation_data=None)

    vgg16.save(fintune_save_path)
    plot_history(history)

if __name__ == "__main__":
    trainfp = open(cfg.train_file_path, 'r')
    train_list = trainfp.readlines()
    trainfp.close()
    testfp = open(cfg.test_file_path, 'r')
    test_list = testfp.readlines()
    testfp.close()
    #pretrain_model(train_list, test_list, optimizer="adam", pretrained_model_path=cfg.pretrain_model_path)
    fintune_model_path = cfg.pretrain_model_path
    fintune_save_path = cfg.fintune_model_path
    fintune_model(train_list, test_list, fintune_model_path, fintune_save_path, optimizer="sgd")