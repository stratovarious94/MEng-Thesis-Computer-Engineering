from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D
from tensorflow.keras.models import Model


class ResNet:
    """
    Extends the final layers of the ResNet model
    """
    def __init__(self, img_dim, num_classes):
        """
        :param img_dim: expected image dimensions (width, height, channels)
        :param num_classes: size of the output vector
        """
        self.img_dim = img_dim
        self.num_classes = num_classes

    def create(self):
        """
        Extends the model's last convolution layer with a GlobalMaxPooling2D layer,
        a Dense layer and a sigmoid layer that produces the output vector with the class probabilities
        :return: Model
        """
        model = resnet50.ResNet50(include_top=False, weights=None, input_shape=self.img_dim, classes=None)
        x = model.output
        x = GlobalMaxPooling2D()(x)
        x = Dense(256, activation='relu', name='final_dense_layer')(x)
        x = Dense(self.num_classes, activation='sigmoid', name='prediction')(x)
        return Model(model.input, x)
