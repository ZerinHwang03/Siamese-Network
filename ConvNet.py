# -*- coding: utf-8 -*-
# CNN模块化
# Layers for CNN

import tensorflow as tf

# 定义卷积层
def conv2d(input_, W_shape, name, reuse=False):
    """
    name - layer name for variable scope W_shape - [height, width, input_layers, output_layers]
    """
    # 在名称空间name下添加变量Variable，但最好传递一个reuse参数。
    with tf.variable_scope(name, reuse=reuse):
        # 对convolution层的权值参数 W 和 b 进行初始化
        W_conv = tf.get_variable('W_conv', shape=W_shape,
                                 initializer=tf.contrib.layers.xavier_initializer())
        b_conv = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]), name="b_conv")

        return tf.nn.relu(tf.nn.conv2d(input_,W_conv,strides=[1, 1, 1, 1],padding='SAME') + b_conv)

# 定义 2x2 max-pooling层
def max_pool_2x2(input_):
    """ Perform max pool with 2x2 kelner"""
    return tf.nn.max_pool(input_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义全连接层
def fc(input_, input_dim, output_dim, name, reuse=False):
    """ Fully connected layer with Sigmoid activation """
    with tf.variable_scope(name, reuse=reuse):
        W_fc = tf.get_variable('W_fc', shape=[input_dim, output_dim],
                        initializer=tf.contrib.layers.xavier_initializer())
        b_fc = tf.Variable(tf.constant(0.1, shape=[output_dim]), name="b_fc")

        return tf.nn.sigmoid(tf.matmul(input_, W_fc) + b_fc)

# 建立CNN模型
# Model creator
def convnet(image):
    """
    Input size: 784
    Image initial size: 28x28x1
    After 5_conv size:  7x7x256
    Output vector of 4096 values
    """
    # reshape中的 -1 可以用于推断应当有的维数，而不是人工进行计算（因为需要保证reshape前后元素总数不变）
    # reshape中shape参数为[-1]时，则表示将传递进来的tensor展平成一维列表
    x_image = tf.reshape(image, [-1, 28, 28, 1])
    conv_1 = conv2d(x_image, [10, 10, 1, 64], "1_conv")
    pool_2 = max_pool_2x2(conv_1)
    conv_3 = conv2d(pool_2, [7, 7, 64, 128], "3_conv")
    pool_4 = max_pool_2x2(conv_3)
    conv_5 = conv2d(pool_4, [4, 4, 128, 256], "5_conv")
    flat_6 = tf.reshape(conv_5, [-1, 7*7*256])
    full_7 = fc(flat_6, 7*7*256, 4096, "7_full")
    return full_7
