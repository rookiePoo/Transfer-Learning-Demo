class config:
    train_file_path = '../data/cat_train.txt'
    test_file_path = '../data/cat_test.txt'
    img_data_dir = '../data'
    pretrain_model_path = '../models/pretrain/vgg16_pretrain'
    fintune_model_path = '../models/fintune/vgg16_fintune'

    #img_size = (224, 224)
    img_size = (48, 48)
    channel_num = 3
    class_num = 12
    batch_size = 8
    learning_rate = 1e-3
    epoches = 3


