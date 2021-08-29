"""Inception_ResNet 2D models in Tensorflow-Keras.
Reference - [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
Inception_ResNet Review: https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc
"""


import tensorflow as tf


def Conv_2D_Block(x, model_width, kernel, strides=(1, 1), padding="same"):
    # 2D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
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


def Inception_ResNet_Module_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception ResNet Module A - Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, 1)

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, 1)
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, 3)

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, 1)
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, 3)
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, 3)

    branch_concat = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl], axis=-1)
    branch1x1_ln = tf.keras.layers.Conv2D(filterB4_1, 1, activation='linear', strides=(1, 1), padding='same', kernel_initializer="he_normal")(branch_concat)

    x = tf.keras.layers.Add(name='Inception_ResNet_Block_A'+str(i))([inputs, branch1x1_ln])
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


def Inception_ResNet_Module_B(inputs, filterB1_1, filterB2_1, filterB2_2, filterB2_3, filterB3_1, i):
    # Inception ResNet Module B - Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, 1)

    branch7x7 = Conv_2D_Block(inputs, filterB2_1, 1)
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (1, 7))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_3, (7, 1))

    branch_concat = tf.keras.layers.concatenate([branch1x1, branch7x7], axis=-1)
    branch1x1_ln = tf.keras.layers.Conv2D(filterB3_1, 1, activation='linear', strides=(1, 1), padding='same', kernel_initializer="he_normal")(branch_concat)

    x = tf.keras.layers.Add(name='Inception_ResNet_Block_B'+str(i))([inputs, branch1x1_ln])
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


def Inception_ResNet_Module_C(inputs, filterB1_1, filterB2_1, filterB2_2, filterB2_3, filterB3_1, i):
    # Inception ResNet Module C - Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, 1)

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, 1)
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, (1, 3))
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_3, (3, 1))

    branch_concat = tf.keras.layers.concatenate([branch1x1, branch3x3], axis=-1)
    branch1x1_ln = tf.keras.layers.Conv2D(filterB3_1, 1, activation='linear', strides=(1, 1), padding='same', kernel_initializer="he_normal")(branch_concat)

    x = tf.keras.layers.Add(name='Inception_ResNet_Block_C'+str(i))([inputs, branch1x1_ln])
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


def Reduction_Block_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB2_3):
    # Reduction Block A
    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(inputs)

    branch3x3 = Conv_2D_Block(inputs, filterB1_1, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, 1)
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, 3)
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, 3, strides=(2, 2), padding='valid')

    x = tf.keras.layers.concatenate([branch_pool, branch3x3, branch3x3dbl], axis=-1, name='Reduction_Block_A')
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


def Reduction_Block_B(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3):
    # Reduction Block B
    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(inputs)

    branch3x3 = Conv_2D_Block(inputs, filterB1_1, 1)
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, 3, strides=(2, 2), padding='valid')

    branch3x3_2 = Conv_2D_Block(inputs, filterB2_1, 1)
    branch3x3_2 = Conv_2D_Block(branch3x3_2, filterB2_2, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, 1)
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, 3)
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, 3, strides=(2, 2), padding='valid')

    x = tf.keras.layers.concatenate([branch_pool, branch3x3, branch3x3_2, branch3x3dbl], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


class Inception_ResNet:
    def __init__(self, length, width, num_channel, num_filters, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False, auxilliary_outputs=False):
        # length: Input Image Length (x-dim)
        # width: Input Image Width (y-dim) [Normally same as the x-dim i.e., Square shape]
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
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.auxilliary_outputs = auxilliary_outputs

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def Inception_ResNet_v1(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, 32, 3, strides=(2, 2), padding='valid')
        x = Conv_2D_Block(x, 32, 3, padding='valid')
        x = Conv_2D_Block(x, 64, 3)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Conv_2D_Block(x, 80, 1)
        x = Conv_2D_Block(x, 192, 3, padding='valid')
        x = Conv_2D_Block(x, 256, 3, strides=(2, 2), padding='valid')

        # 5x Inception ResNet A Blocks - 35 x 35 x 256
        for i in range(5):
            x = Inception_ResNet_Module_A(x, 32, 32, 32, 32, 32, 32, 256, i)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, (1, 1))
            aux_conv = Conv_2D_Block(aux_conv, 768, (5, 5), padding='valid')
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 384, 192, 224, 256)  # Reduction Block A: 17 x 17 x 768

        # 10x Inception ResNet B Blocks - 17 x 17 x 768
        for i in range(10):
            x = Inception_ResNet_Module_B(x, 128, 128, 128, 128, 896, i)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, (1, 1))
            aux_conv = Conv_2D_Block(aux_conv, 768, (5, 5), padding='valid')
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 256, 384, 256, 256, 256, 256, 256)  # Reduction Block B: 8 x 8 x 1280

        # 5x Inception ResNet C Blocks - 8 x 8 x 1280
        for i in range(5):
            x = Inception_ResNet_Module_C(x, 128, 192, 192, 192, 1792, i)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v4')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_ResNet_v1')

        return model

    def Inception_ResNet_v2(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, 32, 3, strides=(2, 2), padding='valid')
        x = Conv_2D_Block(x, 32, 3, padding='valid')
        x = Conv_2D_Block(x, 64, 3)
        #
        branch1 = Conv_2D_Block(x, 96, 3, strides=(2, 2), padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)
        #
        branch1 = Conv_2D_Block(x, 64, 1)
        branch1 = Conv_2D_Block(branch1, 96, 3, padding='valid')
        branch2 = Conv_2D_Block(x, 64, 1)
        branch2 = Conv_2D_Block(branch2, 64, 7)
        branch2 = Conv_2D_Block(branch2, 96, 3, padding='valid')
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)
        #
        branch1 = Conv_2D_Block(x, 192, 3, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=1)

        # 5x Inception ResNet A Blocks - 35 x 35 x 256
        for i in range(10):
            x = Inception_ResNet_Module_A(x, 32, 32, 32, 32, 48, 64, 384, i)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 384, 192, 224, 256)  # Reduction Block A: 17 x 17 x 768

        # 10x Inception ResNet B Blocks - 17 x 17 x 768
        for i in range(20):
            x = Inception_ResNet_Module_B(x, 192, 128, 160, 192, 1152, i)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, 1)
            aux_conv = Conv_2D_Block(aux_conv, 768, 5)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 256, 384, 256, 288, 256, 288, 320)  # Reduction Block B: 8 x 8 x 1280

        # 5x Inception ResNet C Blocks - 8 x 8 x 1280
        for i in range(10):
            x = Inception_ResNet_Module_C(x, 192, 192, 224, 256, 2144, i)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output)
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1])

        return model
