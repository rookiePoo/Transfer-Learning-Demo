#https://blog.csdn.net/wwxy1995/article/details/81370154
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import plot_model
from keras.models import Model

class VGG16_FC:
    """
    定义一个带3层全连接的vgg16模型
    frozen_layers为自选
    使用预训练模型一般分为两步：
        1、载入预训练模型加上全连接层，冻结全连接层以外的所有预训练好的各层参数，只训练全连接层使模型参数收敛
        2、解锁除了全连接层之外的其他层，解锁层数自己定，再进行一次训练，由于这次参数能产生的梯度已经很小了，所以叫做微调
    """
    def __init__(self, vgg16_fc=None, frozen_layers=None, img_size=(224, 224), img_channels=3, class_num=12):
        #VGG16模型，加载预训练权重，include_top=False表示不保留顶层的三个全连接层
        #当你初始化一个预训练模型时，会自动下载权重到 ~/.keras/models/ 目录下。
        self.img_size = img_size
        self.img_channels = img_channels
        self.class_num = class_num
        self.frozen_layers = frozen_layers
        if vgg16_fc:
            self.vgg16_fc = vgg16_fc
        else:
            w, h, c = img_size[0], img_size[1], img_channels
            input = Input(shape=(w, h, c), name='img_input')
            model = VGG16(weights='imagenet', include_top=False, input_tensor=input)
            #print(model.summary())
            plot_model(model, to_file='vgg_without_fc.png')
            # 冻结预训练模型
            for layer in model.layers:
                layer.trainable = False
            # 加上三层全连接
            pool_out = model.output
            pool_out = Flatten()(pool_out)
            class_output = Dense(self.class_num, activation='softmax', name='vgg16_softmax')(Dropout(0.5)(pool_out))
            # fc1 = Dense(256, activation='relu', name='vgg16_fc1')(Dropout(0.5)(pool_out))
            # fc2 = Dense(256, activation='relu', name='vgg16_fc2')(Dropout(0.5)(fc1))
            # class_output = Dense(self.class_num, activation='softmax', name='vgg16_softmax')(fc2)
            self.vgg16_fc = Model(inputs=[input], outputs=[class_output])
            plot_model(self.vgg16_fc, to_file='vgg_with_fc.png')
        # 选择模型微调的冻结层数，vgg16一般是3的倍数
        # 微调相对于预训练来说可以解锁更多层数，根据特定任务以及数据量的情况而定
        # print(model.summary()) vgg16一共18层，微调可以冻结0，3，6，12，15层
        if self.frozen_layers:
            for layer in self.vgg16_fc.layers[:self.frozen_layers]:
                layer.trainable = False
            for layer in self.vgg16_fc.layers[self.frozen_layers:]:
                layer.trainable = True


