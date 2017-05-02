#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import re
import signal
import sys
import argparse
import threading
import time
from random import shuffle
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
from MODEL import model
from MODEL import unet
from PSNR import psnr
from TEST import test_VDSR
import os
from tf_unet import util

# from MODEL_FACTORIZED import model_factorized

DATA_PATH = './data/bp_ang90_snr20_train/'
TEST_DATA_PATH = './data/bp_ang90_snr20_test/'
ckpt_path = './checkpoints/bp_ang90_snr20/VDSR_res_adam4.cpkt'
IMG_SIZE = (256, 256)
TEST_SIZE = (256, 256)
BATCH_SIZE = 4
USE_ADAM_OPT = True
if USE_ADAM_OPT:
    BASE_LR = 0.0001
else:
    BASE_LR = 0.1
LR_RATE = 0.1
LR_STEP_SIZE = 100  # epoch
MAX_EPOCH = 100

USE_QUEUE_LOADING = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_path')
args = parser.parse_args()
model_path = args.model_path


def get_img_list(data_path):
    l = glob.glob(os.path.join(data_path, '*'))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + '_2.mat'):
                train_list.append([f, f[:-4] + '_2.mat', 2])
            if os.path.exists(f[:-4] + '_3.mat'):
                train_list.append([f, f[:-4] + '_3.mat', 3])
            if os.path.exists(f[:-4] + '_4.mat'):
                train_list.append([f, f[:-4] + '_4.mat', 4])
    return train_list


def get_train_list(data_path):
    l = glob.glob(os.path.join(data_path, '*'))
    print len(l)
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    print len(l)
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + '_2.mat'):
                train_list.append([f, f[:-4] + '_2.mat'])
            if os.path.exists(f[:-4] + '_3.mat'):
                train_list.append([f, f[:-4] + '_3.mat'])
            if os.path.exists(f[:-4] + '_4.mat'):
                train_list.append([f, f[:-4] + '_4.mat'])
    return train_list


def get_image_batch(train_list, offset, batch_size):
    target_list = train_list[offset:offset + batch_size]
    input_list = []
    gt_list = []
    cbcr_list = []
    for pair in target_list:
        input_img = scipy.io.loadmat(pair[1])['patch']
        gt_img = scipy.io.loadmat(pair[0])['patch']
        input_list.append(input_img)
        gt_list.append(gt_img)
    input_list = np.array(input_list)
    input_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
    gt_list = np.array(gt_list)
    gt_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
    return (input_list, gt_list, np.array(cbcr_list))


def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset + batch_size]
    print target_list
    input_list = []
    gt_list = []
    cbcr_list = []
    for pair in target_list:
        input_img = scipy.io.loadmat(pair[1])['img_2']
        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        input_list.append(input_img)
        gt_list.append(gt_img)
    input_list = np.array(input_list)
    input_list.resize([batch_size, input_list[0].shape[0],
                      input_list[0].shape[1], 1])
    gt_list = np.array(gt_list)
    gt_list.resize([batch_size, gt_list[0].shape[0],
                   gt_list[0].shape[1], 1])
    return (input_list, gt_list, np.array(cbcr_list))


if __name__ == '__main__':
    train_list = get_train_list(DATA_PATH)

    if not USE_QUEUE_LOADING:
        print 'not use queue loading, just sequential loading...'

        # ## WITHOUT ASYNCHRONOUS DATA LOADING ###

        train_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE,
                IMG_SIZE[0], IMG_SIZE[1], 1))
        train_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE,
                                  IMG_SIZE[0], IMG_SIZE[1], 1))
        test_input = tf.placeholder(tf.float32, shape=(10, TEST_SIZE[0],
                                    TEST_SIZE[1], 1))
    else:

        # ## WITHOUT ASYNCHRONOUS DATA LOADING ###

        print 'use queue loading'

        # ## WITH ASYNCHRONOUS DATA LOADING ###

        train_input_single = tf.placeholder(tf.float32,
                shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
        train_gt_single = tf.placeholder(tf.float32,
                shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
        q = tf.FIFOQueue(10000, [tf.float32, tf.float32],
                         [[IMG_SIZE[0], IMG_SIZE[1], 1], [IMG_SIZE[0],
                         IMG_SIZE[1], 1]])
        enqueue_op = q.enqueue([train_input_single, train_gt_single])

        (train_input, train_gt) = q.dequeue_many(BATCH_SIZE)

        # ## WITH ASYNCHRONOUS DATA LOADING ###

    # shared_model = tf.make_template('shared_model', model)

    with tf.variable_scope('foo'):  # create the first time
        (train_output, weights) = model(train_input)
    with tf.variable_scope('foo', reuse=True):  # create the second time
        (test_output, _) = model(test_input)

    train_res = tf.subtract(train_gt, train_input)
    loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output,
                         train_res)))
    acc = tf.reduce_mean(tf.cast(tf.equal(train_output, train_res),
                         tf.float32))

    # loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(train_output,train_gt)))

    # acc = tf.reduce_mean(tf.cast(tf.equal(train_output, train_gt),tf.float32))

    # for w in weights:
    # ....loss += tf.nn.l2_loss(w)*1e-4

    global_step = tf.Variable(0, trainable=False)

    # learning_rate ....= tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)

    learning_rate = tf.Variable(BASE_LR)

    if USE_ADAM_OPT:
        optimizer = tf.train.AdamOptimizer(learning_rate)  # tf.train.MomentumOptimizer(learning_rate, 0.9)
        opt = optimizer.minimize(loss, global_step=global_step)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

        lr = BASE_LR
        BASE_NORM = 0.1
        tvars = tf.trainable_variables()
        gvs = zip(tf.gradients(loss, tvars), tvars)

        # norm = BASE_NORM*BASE_LR/lr
        # capped_gvs = [(tf.clip_by_norm(grad, norm), var) for grad, var in gvs]

        norm = 0.01
        capped_gvs = [(tf.clip_by_norm(grad, norm), var) for (grad,
                      var) in gvs]
        opt = optimizer.apply_gradients(capped_gvs,
                global_step=global_step)

    saver = tf.train.Saver(weights, max_to_keep=5,
                           write_version=tf.train.SaverDef.V2)

    shuffle(train_list)

    # config = tf.ConfigProto()

    # config.operation_timeout_in_ms=10000

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        if model_path:
            print 'restore model...'
            saver.restore(sess, model_path)
            print 'Done'


        # ## WITH ASYNCHRONOUS DATA LOADING ###

        def load_and_enqueue(
            coord,
            file_list,
            enqueue_op,
            train_input_single,
            train_gt_single,
            idx=0,
            num_thread=1,
            ):

            count = 0
            length = len(file_list)
            try:
                while not coord.should_stop():
                    i = count % length

                    # i = random.randint(0, length-1)

                    input_img = \
                        scipy.io.loadmat(file_list[i][1])['patch'
                            ].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
                    gt_img = scipy.io.loadmat(file_list[i][0])['patch'
                            ].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
                    sess.run(enqueue_op,
                             feed_dict={train_input_single: input_img,
                             train_gt_single: gt_img})
                    count += 1
            except Exception, e:
                print 'stopping...', idx, e


        # ## WITH ASYNCHRONOUS DATA LOADING ###

        threads = []


        def signal_handler(signum, frame):

            # print "stop training, save checkpoint..."
            # saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)

            sess.run(q.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(threads)
            print 'Done'
            sys.exit(1)


        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)

        if USE_QUEUE_LOADING:
            lrr = BASE_LR
            for epoch in xrange(0, MAX_EPOCH):
                if epoch % LR_STEP_SIZE == 0:

                    train_input_single = tf.placeholder(tf.float32,
                            shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
                    train_gt_single = tf.placeholder(tf.float32,
                            shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
                    q = tf.FIFOQueue(1000, [tf.float32, tf.float32],
                            [[IMG_SIZE[0], IMG_SIZE[1], 1],
                            [IMG_SIZE[0], IMG_SIZE[1], 1]])
                    enqueue_op = q.enqueue([train_input_single,
                            train_gt_single])

                    (train_input, train_gt) = q.dequeue_many(BATCH_SIZE)

                    # ## WITH ASYNCHRONOUS DATA LOADING ###

                    (train_output, weights) = shared_model(train_input,
                            is_training=True)
                    loss = \
                        tf.reduce_mean(tf.nn.l2_loss(tf.subtract(train_output,
                            train_gt)))
                    acc = tf.reduce_mean(tf.equal(train_output,
                            train_gt))

                    # for w in weights:
                    # ....loss += tf.nn.l2_loss(w)*1e-4

                    if USE_ADAM_OPT:
                        opt = optimizer.minimize(loss,
                                global_step=global_step)
                    else:

                        lr = BASE_LR
                        BASE_NORM = 0.1
                        tvars = tf.trainable_variables()
                        gvs = zip(tf.gradients(loss, tvars), tvars)

                        # norm = BASE_NORM*BASE_LR/lr
                        # capped_gvs = [(tf.clip_by_norm(grad, norm), var) for grad, var in gvs]

                        norm = 0.01
                        capped_gvs = [(tf.clip_by_norm(grad, norm),
                                var) for (grad, var) in gvs]
                        opt = optimizer.apply_gradients(capped_gvs,
                                global_step=global_step)

                # create threads

                num_thread = 20
                print 'num thread:', len(threads)
                del threads[:]
                coord = tf.train.Coordinator()
                print 'delete threads...'
                print 'num thread:', len(threads)
                for i in range(num_thread):
                    length = len(train_list) / num_thread
                    t = threading.Thread(target=load_and_enqueue, args=(
                        coord,
                        train_list[i * length:(i + 1) * length],
                        enqueue_op,
                        train_input_single,
                        train_gt_single,
                        i,
                        num_thread,
                        ))
                    threads.append(t)
                    t.start()

                for step in range(len(train_list) // BATCH_SIZE):

                    (
                        _,
                        l,
                        accuracy,
                        output,
                        lr,
                        g_step,
                        ) = sess.run([
                        opt,
                        loss,
                        acc,
                        train_output,
                        learning_rate,
                        global_step,
                        ])
                print '[epoch %2.4f] loss %.4f\t acc %.4f\t lr %.7f' \
                    % (epoch + float(step) * BATCH_SIZE
                       / len(train_list), np.sum(l), accuracy, lr)

                # print "[epoch %2.4f] loss %.4f\t lr %.5f\t norm %.2f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr, norm)
                # saver.save(sess, "./checkpoints/VDSR_adam_epoch_%03d.ckpt" % epoch ,global_step=global_step)

                saver.save(sess, ckpt_path)
                if epoch % LR_STEP_SIZE == 19:
                    sess.run(q.close(cancel_pending_enqueues=True))
                    print 'request stop...'
                    coord.request_stop()
                    print 'join threads...'
                    coord.join(threads, stop_grace_period_secs=10)
                    lrr = lrr / 10
        else:
            prediction_path = './prediction_bp_ang90_snr20_VDSR_res'
            prediction_path = os.path.abspath(prediction_path)
            if not os.path.exists(prediction_path):
                os.makedirs(prediction_path)

                # len(train_list) // BATCH_SIZE

            for epoch in xrange(0, MAX_EPOCH):
                for step in range(2000):
                    offset = step * BATCH_SIZE
                    (input_data, gt_data, cbcr_data) = \
                        get_image_batch(train_list, offset, BATCH_SIZE)
                    feed_dict = {train_input: input_data,
                                 train_gt: gt_data}

                    (
                        _,
                        l,
                        accuracy,
                        output,
                        lr,
                        g_step,
                        ) = sess.run([
                        opt,
                        loss,
                        acc,
                        train_output,
                        learning_rate,
                        global_step,
                        ], feed_dict=feed_dict)

                    # del input_data, gt_data, cbcr_data

                print output.shape
                img = util.combine_img_prediction(input_data, gt_data,
                        output + input_data)
                name = 'epoch_%s' % epoch
                util.save_image(img, '%s/%s.jpg' % (prediction_path,
                                name))
                print '[epoch %2.4f] loss %.4f\t acc %.4f\t lr %.7f' \
                    % (epoch + float(step) * BATCH_SIZE
                       / len(train_list), np.sum(l), accuracy, lr)
                psnr_bicub = psnr(input_data, gt_data, 0)
                psnr_vdsr = psnr(output + input_data, gt_data, 0)
                print 'PSNR: bicubic %f\U-NET %f' % (psnr_bicub,
                        psnr_vdsr)

                # print "[epoch %2.4f] loss %.4f\t lr %.7f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr)

                # saver.save(sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)

                saver.save(sess, ckpt_path)

                # test_VDSR(epoch, ckpt_path, TEST_DATA_PATH)

                img_list = get_train_list(TEST_DATA_PATH)

                # print img_list

                (input_list, gt_list, scale_list) = \
                    get_test_image(img_list, 0, 10)
                start_t = time.time()

                feed_dict = {test_input: input_list, train_gt: gt_data,
                             train_input: input_data}

                (
                    _,
                    l,
                    accuracy,
                    output,
                    lr,
                    g_step,
                    ) = sess.run([
                    opt,
                    loss,
                    acc,
                    test_output,
                    learning_rate,
                    global_step,
                    ], feed_dict=feed_dict)
                print 'test_output', output.shape
                end_t = time.time()
                print 'end_t', end_t, 'start_t', start_t
                print 'time consumption', end_t - start_t
                img = util.combine_img_prediction(input_list, gt_list,
                        output + input_list)
                name = 'test_epoch_%s' % epoch
                util.save_image(img, '%s/%s.jpg' % (prediction_path,
                                name))
                print '[test epoch %2.4f] loss %.4f\t acc %.4f\t lr %.7f' \
                    % (epoch + float(step) * BATCH_SIZE
                       / len(train_list), np.sum(l), accuracy, lr)
                psnr_bicub = psnr(input_list, gt_list, 0)
                psnr_vdsr = psnr(output + input_list, gt_list, 0)
                print 'test PSNR: bicubic %f\U-NET %f' % (psnr_bicub,
                        psnr_vdsr)
