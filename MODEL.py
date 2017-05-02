#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import, \
    unicode_literals
import tensorflow as tf
import numpy as np

import os
import shutil
from collections import OrderedDict
import logging
from tf_unet import util
from tf_unet.layers import weight_variable, weight_variable_devonc, \
    bias_variable, conv2d, deconv2d, max_pool, crop_and_concat, \
    pixel_wise_softmax_2, cross_entropy


# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102

def batch_norm_wrapper(inputs, is_training, decay=0.999):

    epsilon = 1e-3
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),
                           trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]),
                          trainable=False)

    if is_training:
        (batch_mean, batch_var) = tf.nn.moments(inputs, [0, 1, 2])

        # Small epsilon value for the BN transform



        # print(batch_mean.get_shape())
        # print(pop_mean.get_shape())

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean
                               * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1
                              - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                inputs,
                batch_mean,
                batch_var,
                beta,
                scale,
                epsilon,
                )
    else:
        return tf.nn.batch_normalization(
            inputs,
            pop_mean,
            pop_var,
            beta,
            scale,
            epsilon,
            )


def unet(
    x,
    is_training,
    keep_prob=1,
    channels=1,
    n_class=1,
    layers=3,
    features_root=64,
    filter_size=3,
    pool_size=2,
    summaries=False,
    ):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    with tf.device('/gpu:0'):
        logging.info('Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}'.format(layers=layers,
                     features=features_root, filter_size=filter_size,
                     pool_size=pool_size))

        # Placeholder for the input image

        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]

        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        in_size = 1000
        size = in_size

        # down layers

        for layer in range(0, layers):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = tf.get_variable('down_conv_00_w1', [filter_size,
                        filter_size, channels, features],
                        initializer=tf.random_normal_initializer(stddev=stddev))
            else:
                w1 = tf.get_variable('down_conv_%02d_w1' % (layer + 1),
                        [filter_size, filter_size, features // 2,
                        features],
                        initializer=tf.random_normal_initializer(stddev=stddev))
            w2 = tf.get_variable('down_conv_%02d_w2' % (layer + 1),
                                 [filter_size, filter_size, features,
                                 features],
                                 initializer=tf.random_normal_initializer(stddev=stddev))
            b1 = tf.get_variable('conv_%02d_b1' % (layer + 1),
                                 [features],
                                 initializer=tf.constant_initializer(0.1))
            b2 = tf.get_variable('conv_%02d_b2' % (layer + 1),
                                 [features],
                                 initializer=tf.constant_initializer(0.1))
            conv1 = conv2d(in_node, w1, keep_prob)
            print(conv1.get_shape())
            conv1 = batch_norm_wrapper(conv1, is_training)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(tmp_h_conv, w2, keep_prob)
            conv2 = batch_norm_wrapper(conv2, is_training)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))
            if layer < layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
        in_node = dw_h_convs[layers - 1]

            # up layers

        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))

             # wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)

            wd = tf.get_variable('up_conv_%02d_wd' % (layer + 1),
                                 [pool_size, pool_size, features // 2,
                                 features],
                                 initializer=tf.random_normal_initializer(stddev=stddev))

            # bd = bias_variable([features//2])

            bd = tf.get_variable('up_conv_%02d_bd' % (layer + 1),
                                 [features // 2],
                                 initializer=tf.constant_initializer(0.1))
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer],
                    h_deconv)
            deconv[layer] = h_deconv_concat

            # w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)

            w1 = tf.get_variable('up_conv_%02d_w1' % (layer + 1),
                                 [filter_size, filter_size, features,
                                 features // 2],
                                 initializer=tf.random_normal_initializer(stddev=stddev))

            # w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)

            w2 = tf.get_variable('up_conv_%02d_w2' % (layer + 1),
                                 [filter_size, filter_size, features
                                 // 2, features // 2],
                                 initializer=tf.random_normal_initializer(stddev=stddev))

            # b1 = bias_variable([features//2])

            b1 = tf.get_variable('up_conv_%02d_b1' % (layer + 1),
                                 [features // 2],
                                 initializer=tf.constant_initializer(0.1))

            # b2 = bias_variable([features//2])

            b2 = tf.get_variable('up_conv_%02d_b2' % (layer + 1),
                                 [features // 2],
                                 initializer=tf.constant_initializer(0.1))

            conv1 = conv2d(h_deconv_concat, w1, keep_prob)
            conv1 = batch_norm_wrapper(conv1, is_training)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(h_conv, w2, keep_prob)
            conv2 = batch_norm_wrapper(conv2, is_training)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            # size *= 2
            # size -= 4

        # Output Map
        # weight = weight_variable([1, 1, features_root, n_class], stddev)

        weight = tf.get_variable('weight', [1, 1, features_root,
                                 n_class],
                                 initializer=tf.random_normal_initializer(stddev=stddev))

        # bias = bias_variable([n_class])

        bias = tf.get_variable('bias', [n_class],
                               initializer=tf.constant_initializer(0.1))
        conv = conv2d(in_node, weight, tf.constant(1.0))

            # conv = batch_norm_wrapper(conv, is_training)

        output_map = tf.nn.relu(conv + bias)

        # output_map = tf.add(output_map, x_image)

        up_h_convs['out'] = output_map

        if summaries:
            for (i, (c1, c2)) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01' % i,
                                 get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i,
                                 get_image_summary(c2))

            for k in pools.keys():
                tf.summary.image('summary_pool_%02d' % k,
                                 get_image_summary(pools[k]))

            for k in deconv.keys():
                tf.summary.image('summary_deconv_concat_%02d' % k,
                                 get_image_summary(deconv[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram('dw_convolution_%02d' % k
                        + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram('up_convolution_%s' % k
                        + '/activations', up_h_convs[k])

        variables = []
        for (w1, w2) in weights:
            variables.append(w1)
            variables.append(w2)

        for (b1, b2) in biases:
            variables.append(b1)
            variables.append(b2)

        # return output_map, variables, int(in_size - size)

            return (output_map, variables)


def model(input_tensor):
    with tf.device('/gpu:0'):
        weights = []
        tensor = None

        # conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())

        conv_00_w = tf.get_variable('conv_00_w', [3, 3, 1, 64],
                                    initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0
                                    / 9)))
        conv_00_b = tf.get_variable('conv_00_b', [64],
                                    initializer=tf.constant_initializer(0))
        weights.append(conv_00_w)
        weights.append(conv_00_b)
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor,
                            conv_00_w, strides=[1, 1, 1, 1],
                            padding='SAME'), conv_00_b))

        for i in range(18):

            # conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())

            conv_w = tf.get_variable('conv_%02d_w' % (i + 1), [3, 3,
                    64, 64],
                    initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0
                    / 9 / 64)))
            conv_b = tf.get_variable('conv_%02d_b' % (i + 1), [64],
                    initializer=tf.constant_initializer(0))
            weights.append(conv_w)
            weights.append(conv_b)
            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor,
                                conv_w, strides=[1, 1, 1, 1],
                                padding='SAME'), conv_b))

        # conv_w = tf.get_variable("conv_19_w", [3,3,64,1], initializer=tf.contrib.layers.xavier_initializer())

        conv_w = tf.get_variable('conv_20_w', [3, 3, 64, 1],
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0
                                 / 9 / 64)))
        conv_b = tf.get_variable('conv_20_b', [1],
                                 initializer=tf.constant_initializer(0))
        weights.append(conv_w)
        weights.append(conv_b)
        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w,
                                strides=[1, 1, 1, 1], padding='SAME'),
                                conv_b)

        tensor = tf.add(tensor, input_tensor)
        return (tensor, weights)
