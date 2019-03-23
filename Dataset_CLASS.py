# -*- coding: utf-8 -*-
# 定义Dataset类
# 用于存储数据并为batches训练和测试提供接口。
# 以1:1的比例生成正负样本。
# 且每个标签的比例取决于labels_equal。

import numpy as np
import time
import random

class Dataset:
    """
    Store data and provide interface for batches training and testing
    Create positive and negative pairs with ratio 1:1
    Ratio of pairs per label depends on labels_equal

    Requires: data, labels (with one hot encoding), number of labels(CLASS), (eqal)
    """

    # 初始化
    def __init__(self, data, labels, n_labels, labels_equal=True, max_pairs=-1):
        self.n_labels = n_labels
        # np.where(np.argmax(labels,1)==i)[0]:
        # np.argmax(labels,1)返回labels在2维视角下最大的索引，应该是一个一维array，元素个数与传入的label个数一致
        # 每个label是one-hot编码。一个label对应一个一维array
        # 当np.where()只有condition时，返回结果为condition中判定对象满足condition的元素索引，同样为一维array
        self.label_indices = [np.where(np.argmax(labels, 1) == i)[0]
                              for i in range(n_labels)]
        self.data = data
        self.epoch = 0
        self.labels_equal = labels_equal
        self.max_pairs = max_pairs
        self.pos_pairs = self.generatePosPairs
        self.neg_pairs = self.generateNegPairs()
        self.length = len(self.pos_pairs)
        self.index = 0

    # 生成正样本对
    @property
    def generatePosPairs(self):
        """ Returns positive pairs created from data set """
        # 初始化正样本列表
        pairs = []
        # labels_len为一维数组，元素为各个类的样本对对数
        labels_len = [len(self.label_indices[d])
                      for d in range(self.n_labels)]


        start_time = time.time() # DEBUG

        if self.labels_equal or self.max_pairs != -1:
            # Number of pairs depends on smallest label dataset
            n = min(labels_len)

            lab = 0
            idx = 0
            pad = 1

            while len(pairs) < self.max_pairs and pad < n:
                pairs += [[self.data[self.label_indices[lab][idx]],
                           self.data[self.label_indices[lab][idx + pad]]]]

                lab = (lab + 1) % self.n_labels
                if lab == 0:
                    idx += 1
                    if (idx + pad) >= n:
                        idx = 0
                        pad += 1

        else:
            # Create maximum number of pairs
            for lab in range(self.n_labels):
                n = labels_len[lab]
                for i in range(n-1):
                    for j in range(i+1, n):
                        pairs += [[self.data[self.label_indices[lab][i]],
                                    self.data[self.label_indices[lab][j]]]]

        print("Positive pairs generated in", time.time() - start_time) # DEBUG
        return np.array(pairs)

    def generateNegPairs(self):
        """ Retruns random negative pairs same length as positive pairs """
        # 初始化负样本对列表
        pairs = []
        #
        chosen = []
        i = 0
        start_time = time.time() # DEBUG
        # 需要让负样本对数和正样本对数一致
        while len(pairs) < len(self.pos_pairs):
            j = (i + random.randrange(1, self.n_labels)) % self.n_labels
            choice = [random.choice(self.label_indices[i]),
                      random.choice(self.label_indices[j])]
            if choice not in chosen:
                chosen += [choice]
                pairs += [[self.data[choice[0]], self.data[choice[1]]]]
            i = (i + 1) % self.n_labels

        print("Negative pairs generated in", time.time() - start_time) # DEBUG
        return np.array(pairs)

    def get_epoch(self):
        """ Get current dataset epoch """
        return self.epoch

    def get_length(self):
        """ Get positive pairs length """
        return self.length

    def next_batch(self, batch_size):
        """
        Returns batch of images and labels of given length
        Requires: even batch size
        """
        start = self.index
        l_size = int(batch_size / 2)
        self.index += l_size

        if self.index > self.length:
            # Shuffle the data
            perm = np.arange(self.length)
            np.random.shuffle(perm)
            self.pos_pairs = self.pos_pairs[perm]
            self.neg_pairs = self.generateNegPairs()
            # Start next epoch
            start = 0
            self.epoch += 1
            self.index = l_size

        end = self.index
        return (np.append(self.pos_pairs[start:end],
                          self.neg_pairs[start:end], 0),
                np.append(np.ones((l_size, 1)),
                          np.zeros((l_size, 1)), 0))

    def random_batch(self, batch_size):
        """
        Returns random randomly shuffled batch - for testing
        *** Maybe not neccesary ***
        """
        pass
    
