# coding=utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim


class FPN(object):
    def __init__(self, num_layer, num_fc, weight_decay, kernel_size, regularizer):
        self.num_layers = num_layer
        self.num_fcs = num_fc

        self.conv_layers = [64, 128, 256, 512]
        self.fc_layers = [4096, 4096, 1000]
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.regularizer = regularizer
        self.endpoints = {}
        self.varlist=[]

    def build_model(self, input):
        net = input
        with slim.arg_scope([slim.conv2d], kernel_size=self.kernel_size,
                            padding="SAME", activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.truncated_normal_initializer,
                            weights_regularizer=slim.l2_regularizer(self.regularizer)
                            ):
            with slim.arg_scope([slim.max_pool2d], padding="VALID", kernel_size=[2, 2]):
                for i in range(self.num_layers):
                    scope_conv = "conv_" + str(i)
                    scope_pool = "pool_" + str(i)
                    net = slim.repeat(net, 3, slim.conv2d, self.conv_layers[i], scope=scope_conv)
                    net = slim.max_pool2d(net, [2, 2], scope=scope_pool)
                    self.endpoints[scope_pool] = net
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                for i in range(self.num_fcs - 1):
                    scope_fc = "fc_" + str(i)
                    scope_dropout = "dropout_" + str(i)
                    net = slim.fully_connected(net, self.fc_layers[i], scope=scope_fc)
                    net = slim.dropout(net, self.dropout, scope=scope_dropout)
            scope_fc = "fc_" + str(self.num_fcs - 1)
            logits = slim.fully_connected(net, self.fc_layers[self.num_fcs - 1], activation_fn=None, scope=scope_fc)
            self.endpoints["logits"] = logits
            self.varlist=tf.trainable_variables()
            # self.varlist=slim.get_trainable_variables()
        return logits, self.endpoints,self.varlist

    def loss(self, pred, label):
        loss = slim.losses.softmax_cross_entropy(pred, label)
        return loss

