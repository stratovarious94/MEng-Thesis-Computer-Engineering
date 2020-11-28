path_to_original_dataset = 'd:/datasets/amazon/train/train_v2.csv'
path_to_train_set = 'data/train'
path_to_valid_set = 'data/valid'
path_to_test_set = 'data/test'
path_to_weights = 'data/weights/'
path_to_predictions = 'data/predictions/'
path_to_history = 'data/history/'
path_to_plots = 'data/plots/'

input_size = 256
batch_size = 16
num_classes = 17
path_to_img_directory = 'd:/datasets/amazon/train/'
color_mode = 'rgb'
channels = 3
img_format = 'jpg'

vit_settings = {
    'patch_size': 16,
    'num_layers': 4,
    'd_model': 1024,
    'num_heads': 8,
    'mlp_dim': 2048,
    'dropout': 0.1
}
