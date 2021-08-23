"""Inception V3 model for Keras.
Reference - [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
Inception_v3 Review: https://sh-tsang.medium.com/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c
Inception_v4 Review: https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc
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

class Inception_ResNet:
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

    def Inception_ResNet_v1(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Stem
        x = Conv_1D_Block(inputs, 32, 3, strides=2, padding='valid')
        x = Conv_1D_Block(x, 32, 3, padding='valid')
        x = Conv_1D_Block(x, 64, 3)
        x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)
        x = Conv_1D_Block(x, 80, 1)
        x = Conv_1D_Block(x, 192, 3, padding='valid')
        x = Conv_1D_Block(x, 256, 3, strides=2, padding='valid')

        # 5x Inception ResNet A Blocks - 35 x 35 x 256
        for i in range(5):
            branch1x1 = Conv_1D_Block(x, 32, 1)

            branch3x3 = Conv_1D_Block(x, 32, 1)
            branch3x3 = Conv_1D_Block(branch3x3, 32, 3)

            branch3x3dbl = Conv_1D_Block(x, 32, 1)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 32, 3)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 32, 3)

            branch_concat = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl], axis=-1)
            branch1x1_ln = tf.keras.layers.Conv1D(256, 1, activation='linear', strides=1, padding='same', kernel_initializer="he_normal")(branch_concat)

            x = tf.keras.layers.Add()([x, branch1x1_ln])
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        # Reduction A: 17 x 17 x 768
        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)

        branch3x3 = Conv_1D_Block(x, 384, 3, strides=2, padding='valid')

        branch3x3dbl = Conv_1D_Block(x, 192, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 224, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 256, 3, strides=2, padding='valid')

        x = tf.keras.layers.concatenate([branch_pool, branch3x3, branch3x3dbl], axis=-1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # 10x Inception ResNet B Blocks - 17 x 17 x 768
        for i in range(10):
            branch1x1 = Conv_1D_Block(x, 128, 1)

            branch7x7 = Conv_1D_Block(x, 128, 1)
            branch7x7 = Conv_1D_Block(branch7x7, 128, 7)

            branch_concat = tf.keras.layers.concatenate([branch1x1, branch7x7], axis=-1)
            branch1x1_ln = tf.keras.layers.Conv1D(896, 1, activation='linear', strides=1, padding='same', kernel_initializer="he_normal")(branch_concat)

            x = tf.keras.layers.Add()([x, branch1x1_ln])
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 128, 1)
            aux_output_1 = self.MLP(aux_conv)

        # Reduction B: 8 x 8 x 1280
        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)

        branch3x3 = Conv_1D_Block(x, 256, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 384, 3, strides=2, padding='valid')

        branch3x3_2 = Conv_1D_Block(x, 256, 1)
        branch3x3_2 = Conv_1D_Block(branch3x3_2, 256, 3, strides=2, padding='valid')

        branch3x3dbl = Conv_1D_Block(x, 256, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 256, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 256, 3, strides=2, padding='valid')

        x = tf.keras.layers.concatenate([branch_pool, branch3x3, branch3x3_2, branch3x3dbl], axis=-1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # 5x Inception ResNet C Blocks - 8 x 8 x 1280
        for i in range(5):
            branch1x1 = Conv_1D_Block(x, 192, 1)

            branch3x3 = Conv_1D_Block(x, 192, 1)
            branch3x3 = Conv_1D_Block(branch3x3, 192, 3)

            branch_concat = tf.keras.layers.concatenate([branch1x1, branch3x3], axis=-1)
            branch1x1_ln = tf.keras.layers.Conv1D(1792, 1, activation='linear', strides=1, padding='same', kernel_initializer="he_normal")(branch_concat)

            x = tf.keras.layers.Add()([x, branch1x1_ln])
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

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

        # 5x Inception ResNet A Blocks - 35 x 35 x 256
        for i in range(10):
            branch1x1 = Conv_1D_Block(x, 32, 1)

            branch3x3 = Conv_1D_Block(x, 32, 1)
            branch3x3 = Conv_1D_Block(branch3x3, 32, 3)

            branch3x3dbl = Conv_1D_Block(x, 32, 1)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 48, 3)
            branch3x3dbl = Conv_1D_Block(branch3x3dbl, 64, 3)

            branch_concat = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl], axis=-1)
            branch1x1_ln = tf.keras.layers.Conv1D(192, 1, activation='linear', strides=1, padding='same', kernel_initializer="he_normal")(branch_concat)

            x = tf.keras.layers.Add()([x, branch1x1_ln])
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        # Reduction A: 17 x 17 x 768
        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)

        branch3x3 = Conv_1D_Block(x, 384, 3, strides=2, padding='valid')

        branch3x3dbl = Conv_1D_Block(x, 192, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 224, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 256, 3, strides=2, padding='valid')

        x = tf.keras.layers.concatenate([branch_pool, branch3x3, branch3x3dbl], axis=-1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # 10x Inception ResNet B Blocks - 17 x 17 x 768
        for i in range(20):
            branch1x1 = Conv_1D_Block(x, 192, 1)

            branch7x7 = Conv_1D_Block(x, 128, 1)
            branch7x7 = Conv_1D_Block(branch7x7, 192, 7)

            branch_concat = tf.keras.layers.concatenate([branch1x1, branch7x7], axis=-1)
            branch1x1_ln = tf.keras.layers.Conv1D(832, 1, activation='linear', strides=1, padding='same', kernel_initializer="he_normal")(branch_concat)

            x = tf.keras.layers.Add()([x, branch1x1_ln])
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling1D(pool_size=5, strides=3, padding='valid')(x)
            aux_conv = Conv_1D_Block(aux_pool, 128, 1)
            aux_conv = Conv_1D_Block(aux_conv, 768, 5)
            aux_output_1 = self.MLP(aux_conv)

        # Reduction B: 8 x 8 x 1280
        branch_pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)

        branch3x3 = Conv_1D_Block(x, 256, 1)
        branch3x3 = Conv_1D_Block(branch3x3, 384, 3, strides=2, padding='valid')

        branch3x3_2 = Conv_1D_Block(x, 256, 1)
        branch3x3_2 = Conv_1D_Block(branch3x3_2, 288, 3, strides=2, padding='valid')

        branch3x3dbl = Conv_1D_Block(x, 256, 1)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 288, 3)
        branch3x3dbl = Conv_1D_Block(branch3x3dbl, 320, 3, strides=2, padding='valid')

        x = tf.keras.layers.concatenate([branch_pool, branch3x3, branch3x3_2, branch3x3dbl], axis=-1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # 5x Inception ResNet C Blocks - 8 x 8 x 1280
        for i in range(10):
            branch1x1 = Conv_1D_Block(x, 192, 1)

            branch3x3 = Conv_1D_Block(x, 192, 1)
            branch3x3 = Conv_1D_Block(branch3x3, 256, 3)

            branch_concat = tf.keras.layers.concatenate([branch1x1, branch3x3], axis=-1)
            branch1x1_ln = tf.keras.layers.Conv1D(1824, 1, activation='linear', strides=1, padding='same', kernel_initializer="he_normal")(branch_concat)

            x = tf.keras.layers.Add()([x, branch1x1_ln])
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v4')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_ResNet_v2')

        return model
