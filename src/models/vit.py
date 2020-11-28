import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.backend import zeros


class MultiHeadSelfAttention:
    """
    Custom attention layer
    """
    def __init__(self, embed_dim, num_heads=8):
        """
        :param embed_dim: int, same as d_model, size of the embedding
        :param num_heads: int, number of self attention blocks from which an image patch will pass
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        # each attention head will take a subset of the embedding
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        # matrix multiplication:
        # [batch_size, num_heads, n_patches+1, d_model//num_heads] @
        # [batch_size, num_heads, n_patches+1, d_model//num_heads] ->
        # [batch_size, num_heads, n_patches+1, num_heads]. For each head how much it aligns with the other heads
        score = tf.matmul(query, key, transpose_b=True)
        # get the shape of the last dimension of key which is: d_model//num_heads
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        # rescale the scores of the matrix multiplication
        scaled_score = score / tf.math.sqrt(dim_key)
        # get the softmax of the alighment between the heads which is a [batch_size, n_patches+1, num_heads] tensor
        weights = tf.nn.softmax(scaled_score, axis=-1)
        # get the output for the calculated attention weights: [batch_size, num_heads, n_patches+1, d_model//num_heads]
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        # reshape the tensor to [batch_size, n_patches+1, num_heads, d_model//num_heads]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        # transpose the tensor to [batch_size, num_heads, n_patches+1, d_model//num_heads]
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def build_MultiHeadSelfAttention(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # pass the embeddings from 3 different dense layers and get [batch_size, n_patches+1, d_model] tensors
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # create a tensor with a subset of the embedding for each
        # head [batch_size, num_heads, n_patches+1, d_model//num_heads]
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # attention [batch_size, num_heads, n_patches+1, d_model//num_heads]
        # weights [batch_size, n_patches+1, num_heads]
        attention, weights = self.attention(query, key, value)
        # attention [batch_size, n_patches+1, num_heads, d_model//num_heads]
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        # concat_attention [batch_size, n_patches+1, d_model]
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        # Feed the result of the attention network to a dense layer to get back [batch_size, n_patches+1, d_model]
        return self.combine_heads(concat_attention)


class TransformerBlock:
    """
    Custom transformer block
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        """
        :param embed_dim: int, same as d_model, size of the embedding
        :param num_heads: int, number of self attention blocks from which an image patch will pass
        :param mlp_dim: int, size of the dense layer before the output layer
        :param dropout: float, dropout rate
        """
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                # change input to [batch_size, n_patches + 1, mlp_dim]
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dropout(dropout),
                # change input to [batch_size, n_patches + 1, d_model]
                Dense(embed_dim),
                Dropout(dropout),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def build_TransformerBlock(self, x):
        inputs_norm = self.layernorm1(x)
        attn_output = self.att.build_MultiHeadSelfAttention(inputs_norm)
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + x

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output)
        # return [batch_size, n_patches + 1, d_model]
        return mlp_output + out1


class VisionTransformer:
    """
    Implementation of the VIT model
    """
    def __init__(self, image_size: int, patch_size: int, num_layers: int, num_classes: int, d_model: int,
                 num_heads: int, mlp_dim: int, channels: int = 3, dropout: float = 0.1):
        """
        :param image_size: int, describes the width and height of a square image
        :param patch_size: int, describes the width and height of a square patch
        :param num_layers: int, how many transformer blocks will we use
        :param num_classes: int, length of the probability list we expect as an output
        :param d_model: int, length of the embedding in which each image patch will be reduced
        :param num_heads: int, number of self attention blocks from which an image patch will pass
        :param mlp_dim: int, size of the dense layer before the output layer
        :param channels: int, RGB, RGBA etc
        :param dropout: float, dropout rate
        """
        # how many patches will cover the image
        num_patches = (image_size // patch_size) ** 2
        # total number of elements that each patch will have
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.input = Input(shape=(image_size, image_size, channels))

        # this is the size of the embedding representation of each patch and the output. d_model dictates vector size
        self.pos_emb = zeros(name="pos_emb", shape=(1, num_patches + 1, d_model))
        self.class_emb = zeros(name="class_emb", shape=(1, 1, d_model))
        # layer before the transformation of the patch to embedding
        self.patch_proj = Dense(d_model)
        # create a number of transformer blocks
        self.enc_layers = [TransformerBlock(d_model, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        # head of the model
        self.mlp_head = tf.keras.Sequential(
            [
                LayerNormalization(epsilon=1e-6),
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dropout(dropout),
                Dense(num_classes, activation=tf.keras.activations.sigmoid),
            ]
        )

    def extract_patches(self, images: tuple):
        """
        Create the actual patches for a batch of images
        :param images:
        :return:
        """
        # get the batch size
        batch_size = tf.shape(images)[0]
        # split each image into as many patches of predetermined size possible
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # each patch is now reshaped to [batch size, channels, patch_dim, patch_dim]
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def build_VisionTransformer(self):
        batch_size = tf.shape(self.input)[0]
        # will return a [batch_size, n_patches, channels, patch_dim, patch_dim] tensor
        patches = self.extract_patches(self.input)
        # will return a [batch_size, n_patches, d_model] tensor
        x = self.patch_proj(patches)

        # create the embedding of the output [batch_size, 1, d_model] tensor
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])

        # concat the dense layer results with the class vector horizontally to make [batch_size, n_patches+1, d_model]
        x = tf.concat([class_emb, x], axis=1)
        # elementwise addition to create a [batch_size, n_patches+1, d_model] tensor
        x = x + self.pos_emb

        # iterate through the transformer layers and connect them. Each one returns a [batch_size, n_patches+1, d_model]
        for layer in self.enc_layers:
            x = layer.build_TransformerBlock(x)

        # extract the class embedding (from the patches): [batch_size, 1, d_model] and get [batch_size, num_classes]
        return Model(self.input, self.mlp_head(x[:, 0]))
