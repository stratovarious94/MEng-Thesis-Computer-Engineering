import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.activations import swish, sigmoid
from tensorflow.keras.layers import Dropout, Input, Conv2D, BatchNormalization, GlobalMaxPooling2D, Dense, Add, \
    Lambda, Multiply, DepthwiseConv2D, Layer, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Initializer


class EfficientNetConvInitializer(Initializer):
    """
    Initializes the variables of Convolution kernels
    """
    def __init__(self):
        super(EfficientNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return K.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


class EfficientNetDenseInitializer(Initializer):
    """
    Initializes the variables of Dense kernels
    """
    def __init__(self):
        super(EfficientNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return K.random_uniform(shape, -init_range, init_range, dtype=dtype)


class DropConnect(Layer):
    """
    Initializes the variables of Dense kernels
    """
    def __init__(self, drop_connect_rate=0., **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):

        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += K.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output

        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = {
            'drop_connect_rate': self.drop_connect_rate,
        }
        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """
    Round number of filters based on depth multiplier
    """
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """
    Round number of filters based on depth multiplier
    """
    multiplier = depth_coefficient

    if not multiplier:
        return repeats

    return int(math.ceil(multiplier * repeats))


def SEBlock(input_filters, se_ratio, expand_ratio):
    """
    Squeeze excitation block
    """
    num_reduced_filters = max(1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio
    spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = Conv2D(num_reduced_filters, kernel_size=[1, 1], strides=[1, 1], kernel_initializer=EfficientNetConvInitializer(), padding='same', use_bias=True)(x)
        x = swish(x)
        x = Conv2D(filters, kernel_size=[1, 1], strides=[1, 1], kernel_initializer=EfficientNetConvInitializer(), padding='same', use_bias=True)(x)
        x = sigmoid(x)
        out = Multiply()([x, inputs])
        return out

    return block


def MBConvBlock(input_filters, output_filters, kernel_size, strides, expand_ratio, se_ratio, id_skip, drop_connect_rate,
                batch_norm_momentum=0.99, batch_norm_epsilon=1e-3):
    """
    Inverted convolution block
    """
    channel_axis = -1

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):

        if expand_ratio != 1:
            x = Conv2D(filters, kernel_size=[1, 1], strides=[1, 1], kernel_initializer=EfficientNetConvInitializer(),
                       padding='same', use_bias=False)(inputs)
            x = BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
            x = swish(x)
        else:
            x = inputs

        x = DepthwiseConv2D([kernel_size, kernel_size], strides=strides, depthwise_initializer=EfficientNetConvInitializer(), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
        x = swish(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio)(x)

        x = Conv2D(output_filters, kernel_size=[1, 1], strides=[1, 1], kernel_initializer=EfficientNetConvInitializer(), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)
                x = Add()([x, inputs])
        return x
    return block


def EfficientNet(input_shape, block_args_list, width_coefficient: float, depth_coefficient: float,
                 classes=1000, dropout_rate=0., drop_connect_rate=0., batch_norm_momentum=0.99, batch_norm_epsilon=1e-3,
                 depth_divisor=8, min_depth=None):
    """
    Base EfficientNet architecture
    Taken and modified from https://github.com/qubvel/efficientnet
    Iteration blocks extend the base network through the coefficients

    :param input_shape: expected image size before turning to input tensor
    :param block_args_list: configuration that controls the layers' initialization properties
    :param width_coefficient: regulates the network's width
    :param depth_coefficient: regulates the network's depth
    :param classes: size of the output vector
    :param dropout_rate: reduce overfitting in the Dense layers
    :param drop_connect_rate: dropout rate at skip connections
    :param batch_norm_momentum: batch normalization property
    :param batch_norm_epsilon: batch normalization property
    """
    # Input part
    channel_axis = -1
    inputs = Input(shape=input_shape)
    x = Conv2D(filters=round_filters(32, width_coefficient, depth_divisor, min_depth), kernel_size=[3, 3],
               strides=[2, 2], kernel_initializer=EfficientNetConvInitializer(), padding='same', use_bias=False)(inputs)
    x = BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
    x = swish(x)

    drop_connect_rate_per_block = drop_connect_rate / float(sum([block_args['num_repeat'] for block_args in block_args_list]))

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        # Update block input and output filters based on depth multiplier.
        block_args['input_filters'] = round_filters(block_args['input_filters'], width_coefficient, depth_divisor, min_depth)
        block_args['output_filters'] = round_filters(block_args['output_filters'], width_coefficient, depth_divisor, min_depth)
        block_args['num_repeat'] = round_repeats(block_args['num_repeat'], depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(block_args['input_filters'], block_args['output_filters'], block_args['kernel_size'],
                        block_args['strides'], block_args['expand_ratio'], block_args['se_ratio'],
                        block_args['identity_skip'], drop_connect_rate_per_block * block_idx,
                        batch_norm_momentum, batch_norm_epsilon)(x)

        if block_args['num_repeat'] > 1:
            block_args['input_filters'] = block_args['output_filters']
            block_args['strides'] = [1, 1]

        for _ in range(block_args['num_repeat'] - 1):
            x = MBConvBlock(block_args['input_filters'], block_args['output_filters'], block_args['kernel_size'],
                            block_args['strides'], block_args['expand_ratio'], block_args['se_ratio'],
                            block_args['identity_skip'], drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon)(x)

    # Head part
    # x = Conv2D(filters=round_filters(640, width_coefficient, depth_coefficient, min_depth), kernel_size=[1, 1],
    #            strides=[2, 2], kernel_initializer=EfficientNetConvInitializer(), padding='same', use_bias=False)(x)
    # x = BatchNormalization(axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
    # x = swish(x)
    x = GlobalMaxPooling2D()(x)
    x = Dense(256, kernel_initializer=EfficientNetDenseInitializer())(x)
    x = swish(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(classes, kernel_initializer=EfficientNetDenseInitializer())(x)
    x = sigmoid(x)

    return Model(inputs, x)


def get_default_block_list():
    """
    Configurations for EfficientNets B0 to B7
    """
    return [
        {'input_filters': 32, 'output_filters': 16, 'kernel_size': 3, 'strides': (1, 1), 'num_repeat': 1, 'se_ratio': 0.25, 'expand_ratio': 1, 'identity_skip': True},
        {'input_filters': 16, 'output_filters': 24, 'kernel_size': 3, 'strides': (2, 2), 'num_repeat': 2, 'se_ratio': 0.25, 'expand_ratio': 6, 'identity_skip': True},
        {'input_filters': 24, 'output_filters': 40, 'kernel_size': 5, 'strides': (2, 2), 'num_repeat': 2, 'se_ratio': 0.25, 'expand_ratio': 6, 'identity_skip': True},
        {'input_filters': 40, 'output_filters': 80, 'kernel_size': 3, 'strides': (2, 2), 'num_repeat': 3, 'se_ratio': 0.25, 'expand_ratio': 6, 'identity_skip': True},
        {'input_filters': 80, 'output_filters': 112, 'kernel_size': 5, 'strides': (1, 1), 'num_repeat': 3, 'se_ratio': 0.25, 'expand_ratio': 6, 'identity_skip': True},
        {'input_filters': 112, 'output_filters': 192, 'kernel_size': 5, 'strides': (2, 2), 'num_repeat': 4, 'se_ratio': 0.25, 'expand_ratio': 6, 'identity_skip': True},
        {'input_filters': 192, 'output_filters': 320, 'kernel_size': 3, 'strides': (1, 1), 'num_repeat': 1, 'se_ratio': 0.25, 'expand_ratio': 6, 'identity_skip': True}
    ]


# initialize EfficientNets B0 to B7 using their respective fixed coefficients
def EfficientNetB0(input_shape=None, classes=17, dropout_rate=0.2, drop_connect_rate=0.):
    return EfficientNet(input_shape, get_default_block_list(), width_coefficient=1.0, depth_coefficient=1.0,
                        classes=classes, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate)


def EfficientNetB1(input_shape=None, classes=17, dropout_rate=0.2, drop_connect_rate=0.):
    return EfficientNet(input_shape, get_default_block_list(), width_coefficient=1.0, depth_coefficient=1.1,
                        classes=classes, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate)


def EfficientNetB2(input_shape=None, classes=17, dropout_rate=0.3, drop_connect_rate=0.):
    return EfficientNet(input_shape, get_default_block_list(), width_coefficient=1.1, depth_coefficient=1.2,
                        classes=classes, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate)


def EfficientNetB3(input_shape=None, classes=17, dropout_rate=0.3, drop_connect_rate=0.):
    return EfficientNet(input_shape, get_default_block_list(), width_coefficient=1.2, depth_coefficient=1.4,
                        classes=classes, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate)


def EfficientNetB4(input_shape=None, classes=17, dropout_rate=0.4, drop_connect_rate=0.):
    return EfficientNet(input_shape, get_default_block_list(), width_coefficient=1.4, depth_coefficient=1.8,
                        classes=classes, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate)


def EfficientNetB5(input_shape=None, classes=17, dropout_rate=0.4, drop_connect_rate=0.):
    return EfficientNet(input_shape, get_default_block_list(), width_coefficient=1.6, depth_coefficient=2.2,
                        classes=classes, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate)


def EfficientNetB6(input_shape=None, classes=17, dropout_rate=0.5, drop_connect_rate=0.):
    return EfficientNet(input_shape, get_default_block_list(), width_coefficient=1.8, depth_coefficient=2.6,
                        classes=classes, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate)


def EfficientNetB7(input_shape=None, classes=17, dropout_rate=0.5, drop_connect_rate=0.):
    return EfficientNet(input_shape, get_default_block_list(), width_coefficient=2.0, depth_coefficient=3.1,
                        classes=classes, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate)
