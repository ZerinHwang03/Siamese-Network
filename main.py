# -*- coding: utf-8 -*-

import tensorflow as tf
# import numpy as np
# import random
import time
import ConvNet as CN
import Dataset_CLASS as dts

# Importing dataset 加载MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 加载训练集和测试集
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# 将数据存入Dataset类实例tr_data(train)和te_data(test)中
# 方便使用Dataset类中的方法对样本数据进行处理
tr_data = dts.Dataset(X_train, y_train, 10, max_pairs=5000)
te_data = dts.Dataset(X_test, y_test, 10, max_pairs=5000)

### 模型
### MODEL

# 使用占位符，确定Siamese Network的输入
# images_L images_R分别表示Siamese Network的左右输入图像
images_1 = tf.placeholder(tf.float32, shape=([None, 784]), name='images_1')
images_2 = tf.placeholder(tf.float32, shape=([None, 784]), name='images_2')
labels = tf.placeholder(tf.float32, shape=([None, 1]), name='labels')

# Siamese Network 但实际上是共享同一个预先训练好的Conv Net
# 所以将左右图片分别经过Conv Net输出
with tf.variable_scope("ConvSiameseNet") as scope:
    # 得到img1和img2经过ConvNet的输出结果model_1和model_2
    model_1 = CN.convnet(images_1)
    scope.reuse_variables()
    # 可以使用tf.get_variable_scope()来获取当前scope，reuse_variables()方法可以设置reuse参数为True
    # tf.get_variable_scope().reuse_variables()
    model_2 = CN.convnet(images_2)

# Combine two outputs by L1 distance
# 两幅图片经过ConvNet后的输出结果（即进行空间转换后的结果）之间的L1距离
# tf.substract()是对应元素之间的减法，shape没有发生改变
# tf.abs()是对元素进行abs，shape没有发生改变
# 所以这里的distance的shape和model_1 model_2一致
distance = tf.abs(tf.subtract(model_1, model_2))

# Final layer with sigmoid
# 应用到SiamFC中，训练过程发生改变：ConvNet是不需要训练的。真正需要训练的是ConvNet输出后的相似性的度量
# 但在这个Siamese Network中是不同的，直接对Conv进行训练

# W_out b_out为ConvNet层外结构的参数
W_out = tf.get_variable('W_out', shape=[4096, 1],  initializer=tf.contrib.layers.xavier_initializer())
b_out = tf.Variable(tf.constant(0.1, shape=[1]), name="b_out")

# Output - result of sigmoid - for future use
# Prediction - rounded sigmoid to 0 or 1
# 最终输出结果
output = tf.nn.sigmoid(tf.matmul(distance, W_out) + b_out)

# 将Tensor的值四舍五入为最接近的整数
prediction = tf.round(output)

# Using cross entropy for sigmoid as loss
# @TODO add regularization
# 使用交叉熵定义损失函数
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output))


# Optimizer
# 注意，由于这里的计算图的定义，会导致这里的优化过程会对ConvNet中的权值也进行更新
optimizer = tf.train.AdamOptimizer(learning_rate=0.0004).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0004).minimize(loss)
# Measuring accuracy of model
correct_prediction = tf.equal(prediction, labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### 训练
### TRAINING
batch_size = 64
# batch_size = 128

# 创建会话，并开始对计算图进行操作

with tf.Session() as sess:
    print("Starting training")
    tf.global_variables_initializer().run()

    # Training cycle
    for epoch in range(100):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = 20  # Not accurate
        start_time = time.time()

        # Loop over all batches
        for i in range(total_batch):
            # Fit training using batch data
            tr_input, y = tr_data.next_batch(batch_size)
            _, loss_value, acc = sess.run([optimizer, loss, accuracy],
                                          feed_dict={images_1: tr_input[:, 0],
                                                     images_2: tr_input[:, 1],
                                                     labels: y})
            avg_loss += loss_value
            avg_acc += acc * 100

        duration = time.time() - start_time
        print('epoch %d  time: %f loss %0.5f acc %0.2f' % (epoch,
                                                           duration,
                                                           avg_loss / total_batch,
                                                           avg_acc / total_batch))
        te_pairs, te_y = te_data.next_batch(1000)
        te_acc = accuracy.eval(feed_dict={images_1: te_pairs[:, 0],
                                          images_2: te_pairs[:, 1],
                                          labels: te_y})
        print('Accuracy on test set %0.2f' % (100 * te_acc))

    # 最终测试
    # Final Testing
    tr_pairs, tr_y = te_data.next_batch(1000)
    tr_acc = accuracy.eval(feed_dict={images_1: tr_pairs[:, 0],
                                      images_2: tr_pairs[:, 1],
                                      labels: tr_y})
    print('Accuract training set %0.2f' % (100 * tr_acc))

    te_pairs, te_y = te_data.next_batch(1000)
    te_acc = accuracy.eval(feed_dict={images_1: te_pairs[:, 0],
                                      images_2: te_pairs[:, 1],
                                      labels: te_y})
    print('Accuract test set %0.2f' % (100 * te_acc))

    # TODO Predicting correct label based accuracy on sample of labeled data.
