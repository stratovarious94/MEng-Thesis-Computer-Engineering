# paths to project data directories
path_to_original_dataset = 'd:/datasets/amazon/train/train_v2.csv'
path_to_train_set = 'data/train'
path_to_valid_set = 'data/valid'
path_to_test_set = 'data/test'
path_to_weights = 'data/weights/'
path_to_predictions = 'data/predictions/'
path_to_history = 'data/history/'
path_to_plots = 'data/plots/'
path_to_explainable = 'data/explainable/'

# general settings for all the models
input_size = 256
batch_size = 16
num_classes = 17
path_to_img_directory = 'd:/datasets/amazon/train/'
color_mode = 'rgb'
channels = 3
img_format = 'jpg'

# vision transformer exclusive settings
vit_settings = {
    'patch_size': 16,
    'num_layers': 4,
    'd_model': 1024,
    'num_heads': 8,
    'mlp_dim': 2048,
    'dropout': 0.1
}

# settings for initializing the models in the context of Explainable AI
explainable_cnn_settings = {
    'vgg16': {
        'channels': 4,
        'input_size': 256,
        'ex_format': 'tif',
        'last_conv_layer_name1': 'block5_conv3',
        'last_conv_layer_name2': 'block5_conv3',
        'last_conv_layer_filter_number': 1024
    },
    'resnet': {
        'channels': 3,
        'input_size': 64,
        'ex_format': 'jpg',
        'last_conv_layer_name1': 'conv5_block3_out',
        'last_conv_layer_name2': 'conv5_block3_out',
        'last_conv_layer_filter_number': 2048
    },
    'densenet': {
        'channels': 3,
        'input_size': 64,
        'ex_format': 'jpg',
        'last_conv_layer_name1': 'relu',
        'last_conv_layer_name2': 'relu',
        'last_conv_layer_filter_number': 1024
    },
    'mobilenet': {
        'channels': 3,
        'input_size': 64,
        'ex_format': 'jpg',
        'last_conv_layer_name1': 'conv_pw_13_relu',
        'last_conv_layer_name2': 'conv_pw_13_relu',
        'last_conv_layer_filter_number': 1024
    },
    'efficientnet': {
        'channels': 3,
        'input_size': 256,
        'ex_format': 'jpg',
        'last_conv_layer_name1': 'conv2d_63',
        'last_conv_layer_name2': 'conv2d_127',
        'last_conv_layer_filter_number': 320
    },
    'vit': {
        'channels': 3,
        'input_size': 256,
        'ex_format': 'jpg',
        'last_conv_layer_name1': 'tf_op_layer_add_8',
        'last_conv_layer_name2': 'tf_op_layer_add_17',
        'last_conv_layer_filter_number': 1024
    }
}
