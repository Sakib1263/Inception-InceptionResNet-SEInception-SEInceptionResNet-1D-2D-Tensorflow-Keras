"""Inception 1D_CNN Models in Tensorflow-Keras.
References -
Inception_v1 (GoogLeNet): https://arxiv.org/abs/1409.4842 [Going Deeper with Convolutions]
Inception_v2: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v3: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v4: https://arxiv.org/abs/1602.07261 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning]
"""


import tensorflow as tf


def Conv_1D_Block(x, model_width, kernel, strides=1, padding="same"):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)
    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs         : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)
    return out


class Inception:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False, auxilliary_outputs=False):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Model
        # kernel_size: Kernel or Filter Size of the Input Convolutional Layer
        # num_channel: Number of Channels of the Input Predictor Signals
        # problem_type: Regression or Classification
        # output_nums: Number of Output Classes in Classification mode and output features in Regression mode
        # pooling: Choose either 'max' for MaxPooling or 'avg' for Averagepooling
        # dropout_rate: If turned on, some layers will be dropped out randomly based on the selected proportion
        # auxilliary_outputs: Two extra Auxullary outputs for the Inception models, acting like Deep Supervision
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.auxilliary_outputs = auxilliary_outputs

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def Inception_v1(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Stem
        x = Conv_1D_Block(inputs, self.num_filters, 7, strides=2)
        x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)
        x = Conv_1D_Block(x, self.num_filters, 1, padding='valid')
        x = Conv_1D_Block(x, self.num_filters * 3, 3)
        x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)

        # Inception Block 1
        branch1x1 = Conv_1D_Block(x, 64, 1, padding='valid')

        branch3x3 = Conv_1D_Block(x, 96, 1, padding='valid')
        branch3x3 = Conv_1D_Block(branch3x3, 128, 3)

        branch5x5 = Conv_1D_Block(x, 16, 1, padding='valid')
        branch5x5 = Conv_1D_Block(branch5x5, 32, 5)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 32, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_1')

        # Inception Block 2
        branch1x1 = Conv_1D_Block(x, 128, 1, padding='valid')

        branch3x3 = Conv_1D_Block(x, 128, 1, padding='valid')
        branch3x3 = Conv_1D_Block(branch3x3, 192, 3)

        branch5x5 = Conv_1D_Block(x, 32, 1, padding='valid')
        branch5x5 = Conv_1D_Block(branch5x5, 96, 5)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 64, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_2')

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)

        # Inception Block 3
        branch1x1 = Conv_1D_Block(x, 192, 1, padding='valid')

        branch3x3 = Conv_1D_Block(x, 96, 1, padding='valid')
        branch3x3 = Conv_1D_Block(branch3x3, 208, 3)

        branch5x5 = Conv_1D_Block(x, 16, 1, padding='valid')
        branch5x5 = Conv_1D_Block(branch5x5, 48, 5)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 64, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_3')

        # Inception Block 4
        branch1x1 = Conv_1D_Block(x, 160, 1, padding='valid')

        branch3x3 = Conv_1D_Block(x, 112, 1, padding='valid')
        branch3x3 = Conv_1D_Block(branch3x3, 224, 3)

        branch5x5 = Conv_1D_Block(x, 24, 1, padding='valid')
        branch5x5 = Conv_1D_Block(branch5x5, 64, 5)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 64, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_4')

        # Inception Block 5
        branch1x1 = Conv_1D_Block(x, 128, 1, padding='valid')

        branch3x3 = Conv_1D_Block(x, 128, 1, padding='valid')
        branch3x3 = Conv_1D_Block(branch3x3, 256, 3)

        branch5x5 = Conv_1D_Block(x, 24, 1, padding='valid')
        branch5x5 = Conv_1D_Block(branch5x5, 64, 5)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 64, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_5')

        # Inception Block 6
        branch1x1 = Conv_1D_Block(x, 112, 1, padding='valid')

        branch3x3 = Conv_1D_Block(x, 144, 1, padding='valid')
        branch3x3 = Conv_1D_Block(branch3x3, 288, 3)

        branch5x5 = Conv_1D_Block(x, 32, 1, padding='valid')
        branch5x5 = Conv_1D_Block(branch5x5, 64, 5)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 64, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_6')

        # Inception Block 7
        branch1x1 = Conv_1D_Block(x, 256, 1, padding='valid')

        branch3x3 = Conv_1D_Block(x, 160, 1, padding='valid')
        branch3x3 = Conv_1D_Block(branch3x3, 320, 3)

        branch5x5 = Conv_1D_Block(x, 32, 1, padding='valid')
        branch5x5 = Conv_1D_Block(branch5x5, 128, 5)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 128, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_7')

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 64, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)

        # Inception Block 8
        branch1x1 = Conv_1D_Block(x, 256, 1, padding='valid')

        branch3x3 = Conv_1D_Block(x, 160, 1, padding='valid')
        branch3x3 = Conv_1D_Block(branch3x3, 320, 3)

        branch5x5 = Conv_1D_Block(x, 32, 1, padding='valid')
        branch5x5 = Conv_1D_Block(branch5x5, 128, 5)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 128, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_8')

        # Inception Block 9
        branch1x1 = Conv_1D_Block(x, 384, 1, padding='valid')

        branch3x3 = Conv_1D_Block(x, 192, 1, padding='valid')
        branch3x3 = Conv_1D_Block(branch3x3, 384, 3)

        branch5x5 = Conv_1D_Block(x, 48, 1, padding='valid')
        branch5x5 = Conv_1D_Block(branch5x5, 128, 5)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 128, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_9')

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v1')

        return model

    def Inception_v2(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Stem: 56 x 64
        x = tf.keras.layers.SeparableConv1D(self.num_filters, kernel_size=7, strides=2, depth_multiplier=1, padding='same')(inputs)
        x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)
        x = Conv_1D_Block(x, self.num_filters * 2, 1, padding='valid')
        x = Conv_1D_Block(x, self.num_filters * 6, 3, padding='valid')
        x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)

        # Inception Block 1: 28 x 192
        branch1x1 = Conv_1D_Block(x, 64, 1)

        branch3x3 = Conv_1D_Block(x, 64, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 64, 3)

        branch3x3dbl = Conv_1D_Block(x, 64, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 32, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_1')

        # Inception Block 2: 28 x 256
        branch1x1 = Conv_1D_Block(x, 64, 1)

        branch3x3 = Conv_1D_Block(x, 64, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 96, 3)

        branch3x3dbl = Conv_1D_Block(x, 64, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 64, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_2')

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        # Reduction Block A: 28 x 320
        branch3x3 = Conv_1D_Block(x, 128, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 160, 3, strides=2)

        branch3x3dbl = Conv_1D_Block(x, 64, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3, strides=2)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_A')

        # Inception Block 3: 14 x 576
        branch1x1 = Conv_1D_Block(x, 224, 1)

        branch3x3 = Conv_1D_Block(x, 64, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 96, 3)

        branch3x3dbl = Conv_1D_Block(x, 96, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 128, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 128, 3)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 128, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_3')

        # Inception Block 4: 14 x 576
        branch1x1 = Conv_1D_Block(x, 192, 1)

        branch3x3 = Conv_1D_Block(x, 96, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 128, 3)

        branch3x3dbl = Conv_1D_Block(x, 96, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 128, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 128, 3)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 128, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_4')

        # Inception Block 5: 14 x 576
        branch1x1 = Conv_1D_Block(x, 160, 1)

        branch3x3 = Conv_1D_Block(x, 128, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 160, 3)

        branch3x3dbl = Conv_1D_Block(x, 128, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 160, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 160, 3)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 96, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_5')

        # Inception Block 6: 14 x 576
        branch1x1 = Conv_1D_Block(x, 96, 1)

        branch3x3 = Conv_1D_Block(x, 128, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 192, 3)

        branch3x3dbl = Conv_1D_Block(x, 160, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 192, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 192, 3)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 96, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_6')

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 192, 1)
            aux_output_1 = self.MLP(aux_conv)

        # Reduction Block B: 14 x 576
        branch3x3 = Conv_1D_Block(x, 128, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 192, 3, strides=2)

        branch3x3dbl = Conv_1D_Block(x, 192, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 256, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 256, 3, strides=2)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_B')

        # Inception Block 7: 7 x 1024
        branch1x1 = Conv_1D_Block(x, 352, 1)

        branch3x3 = Conv_1D_Block(x, 192, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 320, 3)

        branch3x3dbl = Conv_1D_Block(x, 160, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 224, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 224, 3)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 128, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_7')

        # Inception Block 8: 7 x 1024
        branch1x1 = Conv_1D_Block(x, 352, 1)

        branch3x3 = Conv_1D_Block(x, 192, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 320, 3)

        branch3x3dbl = Conv_1D_Block(x, 192, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 224, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 224, 3)

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 128, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_8')

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v2')

        return model

    def Inception_v3(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Stem
        x = Conv_1D_Block(inputs, self.num_filters, 3, strides=2, padding='valid')
        x = Conv_1D_Block(x, self.num_filters, 3, padding='valid')
        x = Conv_1D_Block(x, self.num_filters * 2, 3)
        x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)

        x = Conv_1D_Block(x, self.num_filters * 2.5, 1, padding='valid')
        x = Conv_1D_Block(x, self.num_filters * 6, 3, padding='valid')
        x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)

        # 3x Inception-A Blocks: 35 x 256
        branch1x1 = Conv_1D_Block(x, 64, 1)

        branch5x5 = Conv_1D_Block(x, 48, 1)
        branch5x5 = Conv_1D_Block(branch5x5, 64, 5)

        branch3x3dbl = Conv_1D_Block(x, 64, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 32, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='Inception_A1')

        for i in range(2,4):
            branch1x1 = Conv_1D_Block(x, 64, 1)

            branch5x5 = Conv_1D_Block(x, 48, 1)
            branch5x5 = Conv_1D_Block(branch5x5, 64, 5)

            branch3x3dbl = Conv_1D_Block(x, 64, 1)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)

            branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
            branch_pool = Conv_1D_Block(branch_pool, 64, 1)
            x = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='Inception_A'+str(i))

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        # Reduction A: 17 x 768
        branch3x3 = Conv_1D_Block(x, 64, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 384, 3, strides=2, padding='valid')

        branch3x3dbl = Conv_1D_Block(x, 64, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3, strides=2, padding='valid')

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)
        x = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_A')

        # 4x Inception-B Blocks: 17 x 768
        branch1x1 = Conv_1D_Block(x, 192, 1)

        branch7x7 = Conv_1D_Block(x, 128, 1)
        branch7x7 = Conv_1D_Block(branch7x7, 192, 7)

        branch7x7dbl = Conv_1D_Block(x, 128, 1)
        branch7x7dbl = Conv_1D_Block(branch7x7dbl, 128, 7)
        branch7x7dbl = Conv_1D_Block(branch7x7dbl, 192, 7)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 192, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='Inception_B1')

        for i in range(2, 4):
            branch1x1 = Conv_1D_Block(x, 192, 1)

            branch7x7 = Conv_1D_Block(x, 160, 1)
            branch7x7 = Conv_1D_Block(branch7x7, 192, 7)

            branch7x7dbl = Conv_1D_Block(x, 160, 1)
            branch7x7dbl = Conv_1D_Block(branch7x7dbl, 160, 7)
            branch7x7dbl = Conv_1D_Block(branch7x7dbl, 192, 7)

            branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
            branch_pool = Conv_1D_Block(branch_pool, 192, 1)
            x = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='Inception_B'+str(i))

        branch1x1 = Conv_1D_Block(x, 192, 1)

        branch7x7 = Conv_1D_Block(x, 192, 1)
        branch7x7 = Conv_1D_Block(branch7x7, 192, 7)

        branch7x7dbl = Conv_1D_Block(x, 192, 1)
        branch7x7dbl = Conv_1D_Block(branch7x7dbl, 192, 7)
        branch7x7dbl = Conv_1D_Block(branch7x7dbl, 192, 7)

        branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool, 192, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='Inception_B4')

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 192, 1)
            aux_output_1 = self.MLP(aux_conv)

        # Reduction B: 8 x 1280
        branch3x3 = Conv_1D_Block(x, 192, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 320, 3, strides=2, padding='valid')

        branch7x7x3 = Conv_1D_Block(x, 192, 1)
        branch7x7x3 = Conv_1D_Block(branch7x7x3, 192, 7)
        branch7x7x3 = Conv_1D_Block(branch7x7x3, 192, 3, strides=2, padding='valid')

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)
        x = tf.keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=-1, name='Reduction_B')

        # 2x Inception-C Blocks: 8 x 2048
        for i in range(2):
            branch1x1 = Conv_1D_Block(x, 320, 1)

            branch3x3 = Conv_1D_Block(x, 384, 1)
            branch3x3 = Conv_1D_Block(branch3x3, 384, 3)

            branch3x3dbl = Conv_1D_Block(x, 448, 1)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 384, 3)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 384, 3)

            branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
            branch_pool = Conv_1D_Block(branch_pool, 192, 1)
            x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_C'+str(i))

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v3')

        return model

    def Inception_v4(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Stem
        x = Conv_1D_Block(inputs, 32, 3, strides=2, padding='valid')
        x = Conv_1D_Block(x, 32, 3, padding='valid')
        x = Conv_1D_Block(x, 64, 3)
        #
        branch1 = Conv_1D_Block(x, 96, 3, strides=2, padding='valid')
        branch2 = tf.keras.layers.MaxPooling1D(3, strides=2)(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)
        #
        branch1 = Conv_1D_Block(x, 64, 1)
        branch1 = Conv_1D_Block(branch1, 96, 3, padding='valid')
        branch2 = Conv_1D_Block(x, 64, 1)
        branch2 = Conv_1D_Block(branch2, 64, 7)
        branch2 = Conv_1D_Block(branch2, 96, 3, padding='valid')
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)
        #
        branch1 = Conv_1D_Block(x, 192, 3, padding='valid')
        branch2 = tf.keras.layers.MaxPooling1D(3, strides=2)(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=1)

        # 4x Inception-A Blocks - 35 x 256
        for i in range(4):
            branch1x1 = Conv_1D_Block(x, 96, 1)

            branch5x5 = Conv_1D_Block(x, 64, 1)
            branch5x5 = Conv_1D_Block(branch5x5, 96, 3)

            branch3x3dbl = Conv_1D_Block(x, 64, 1)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 96, 3)

            branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
            branch_pool = Conv_1D_Block(branch_pool, 96, 1)
            x = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='Inception_A' + str(i))

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        # Reduction A: 17 x 768
        branch3x3 = Conv_1D_Block(x, 64, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 384, 3, strides=2, padding='valid')

        branch3x3dbl = Conv_1D_Block(x, 192, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 224, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 256, 3, strides=2, padding='valid')

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)
        x = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_A')

        # 7x Inception-B Blocks - 17 x 768
        for i in range(7):
            branch1x1 = Conv_1D_Block(x, 384, 1)

            branch7x7 = Conv_1D_Block(x, 192, 1)
            branch7x7 = Conv_1D_Block(branch7x7, 256, 7)

            branch7x7dbl = Conv_1D_Block(x, 192, 1)
            branch7x7dbl = Conv_1D_Block(branch7x7dbl, 224, 7)
            branch7x7dbl = Conv_1D_Block(branch7x7dbl, 256, 7)

            branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
            branch_pool = Conv_1D_Block(branch_pool, 128, 1)
            x = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='Inception_B' + str(i))

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 128, 1)
            aux_output_1 = self.MLP(aux_conv)

        # Reduction B: 8 x 1280
        branch3x3 = Conv_1D_Block(x, 192, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 192, 3, strides=2, padding='valid')

        branch7x7x3 = Conv_1D_Block(x, 256, 1)
        branch7x7x3 = Conv_1D_Block(branch7x7x3, 320, 7)
        branch7x7x3 = Conv_1D_Block(branch7x7x3, 320, 3, strides=2, padding='valid')

        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)
        x = tf.keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=-1, name='Reduction_B')

        # 3x Inception-C Blocks: 8 x 2048
        for i in range(3):
            branch1x1 = Conv_1D_Block(x, 256, 1)

            branch3x3 = Conv_1D_Block(x, 384, 1)
            branch3x3 = Conv_1D_Block(branch3x3, 512, 3)

            branch3x3dbl = Conv_1D_Block(x, 384, 1)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 512, 3)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 512, 3)

            branch_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
            branch_pool = Conv_1D_Block(branch_pool, 256, 1)
            x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_C' + str(i))

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v4')
        if self.auxilliary_outputs:
            model = tf.keras.layers.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v4')

        return model
