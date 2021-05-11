'''
    File name: CIFAR100-DenseNet201.py
    Author: Yue Liang
    Date last modified: 12/20/2019
    Python Version: 3.7
    TensorFlow 2.1
'''

import tensorflow as tf
import numpy as np 
import argparse
import random
import math
import os

from model.densenet import densenet201


from utils2 import compute_mean_var, norm_images, unpickle, generate_tfrecord, norm_images_using_mean_var, lr_schedule_200ep, lr_schedule_300ep


def parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)

    img = tf.io.decode_raw(features['image_raw'], tf.float32)
    img = tf.reshape(img, shape=(32, 32, 3))

    img = tf.pad(img, [[4, 4], [4, 4], [0, 0]])
    img = tf.image.random_crop(img, [32, 32, 3])
    # img = tf.image.random_flip_left_right(img)

    flip = random.getrandbits(1)
    if flip:
        img = img[:, ::-1, :]


    label = tf.cast(features['label'], tf.int64)
    return img, label

def parse_test(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    img = tf.decode_raw(features['image_raw'], tf.float32)
    img = tf.reshape(img, shape=(32, 32, 3))

    label = tf.cast(features['label'], tf.int64)
    return img, label

def lr_schedule(epoch, tot_ep):
    if tot_ep == 200:
        return lr_schedule_200ep(epoch)
    if tot_ep == 300:
        return lr_schedule_300ep(epoch)
    print('*** Choose correct ep 200 or 300.')

    

def train(batch_size, epoch, network, opt, train_path, test_path):

    train = unpickle(train_path)
    test = unpickle(test_path)
    train_data = train[b'data']
    test_data  = test[b'data']

    x_train = train_data.reshape(train_data.shape[0], 3, 32, 32)
    x_train = x_train.transpose(0, 2, 3, 1)
    y_train = train[b'fine_labels']


    x_test = test_data.reshape(test_data.shape[0], 3, 32, 32)
    x_test = x_test.transpose(0, 2, 3, 1)
    y_test= test[b'fine_labels']

    x_train = norm_images(x_train)
    x_test = norm_images(x_test)

    print('-------------------------------')
    print('--train/test len: ', len(train_data), len(test_data))
    print('--x_train norm: ', compute_mean_var(x_train))
    print('--x_test norm: ', compute_mean_var(x_test))
    print('--batch_size: ', batch_size)
    print('--epoch: ', epoch)
    print('--network: ', network)
    print('--opt: ', opt)
    print('-------------------------------')

    if not os.path.exists('./trans/tran.tfrecords'):
        generate_tfrecord(x_train, y_train, './trans/', 'tran.tfrecords')
        generate_tfrecord(x_test, y_test, './trans/', 'test.tfrecords')

    dataset = tf.data.TFRecordDataset('./trans/tran.tfrecords')
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(batch_size)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)


    next_element = iterator.get_next() 

    x_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_input = tf.placeholder(tf.int64, [None, ])
    y_input_one_hot = tf.one_hot(y_input, 100)
    lr = tf.placeholder(tf.float32, [])

    prob = densenet201(x_input, reuse=False, is_training=True, kernel_initializer=tf.orthogonal_initializer())




    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prob, labels=y_input_one_hot))

    conv_var = [var for var in tf.trainable_variables() if 'conv' in var.name]
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in conv_var])
    loss = l2_loss * 5e-4 + loss

    if opt == 'adam':
        opt = tf.train.AdamOptimizer(lr)
    elif opt == 'momentum':
        opt = tf.train.MomentumOptimizer(lr, 0.9)
    elif opt == 'nesterov':
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss)

    logit_softmax = tf.nn.softmax(prob)
    acc  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit_softmax, 1), y_input), tf.float32))

    #-------------------------------Test-----------------------------------------
    if not os.path.exists('./trans/tran.tfrecords'):
        generate_tfrecord(x_test, y_test, './trans/', 'test.tfrecords')
    dataset_test = tf.data.TFRecordDataset('./trans/test.tfrecords')
    dataset_test = dataset_test.map(parse_test)
    dataset_test = dataset_test.shuffle(buffer_size=10000)
    dataset_test = dataset_test.batch(128)
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next() 


    prob_test = densenet201(x_input, is_training=False, reuse=True, kernel_initializer=None)



    logit_softmax_test = tf.nn.softmax(prob_test)
    acc_test = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logit_softmax_test, 1), y_input), tf.float32))
    #----------------------------------------------------------------------------
    saver = tf.train.Saver(max_to_keep=1, var_list=tf.global_variables())
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    now_lr = 0.001    # Warm Up
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        counter = 0
        max_test_acc = -1
        for i in range(epoch):
            sess.run(iterator.initializer)
            while True:
                try:
                    batch_train, label_train = sess.run(next_element)
                    _, loss_val, acc_val, lr_val= sess.run([train_op, loss, acc, lr], feed_dict={x_input: batch_train, y_input: label_train, lr: now_lr})

                    counter += 1

                    if counter % 100 == 0:
                        print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val)
                    if counter % 1000 == 0:
                        print('start test ')
                        sess.run(iterator_test.initializer)
                        avg_acc = []
                        while True:
                            try:
                                batch_test, label_test = sess.run(next_element_test)
                                acc_test_val = sess.run(acc_test, feed_dict={x_input: batch_test, y_input: label_test})
                                avg_acc.append(acc_test_val)
                            except tf.errors.OutOfRangeError:
                                print('end test ', np.sum(avg_acc)/len(y_test))
                                now_test_acc = np.sum(avg_acc)/len(y_test)
                                if now_test_acc > max_test_acc:
                                    print('***** Max test changed: ', now_test_acc)
                                    max_test_acc = now_test_acc
                                    filename = 'params/distinct/'+'densenet201'+'_{}.ckpt'.format(counter)
                                    saver.save(sess, filename)
                                break
                except tf.errors.OutOfRangeError:
                    print('end epoch %d/%d , lr: %f'%(i, epoch, lr_val))
                    now_lr = lr_schedule(i, epoch)
                    break

# def test(args):
def test(network,test_path,ckpt):
    # train = unpickle('/data/ChuyuanXiong/up/cifar-100-python/train')
    # train_data = train[b'data']
    # x_train = train_data.reshape(train_data.shape[0], 3, 32, 32)
    # x_train = x_train.transpose(0, 2, 3, 1)

    test = unpickle(test_path)
    test_data  = test[b'data']

    x_test = test_data.reshape(test_data.shape[0], 3, 32, 32)
    x_test = x_test.transpose(0, 2, 3, 1)
    y_test= test[b'fine_labels']

    x_test = norm_images(x_test)
    # x_test = norm_images_using_mean_var(x_test, *compute_mean_var(x_train))

    network = network
    ckpt = ckpt

    x_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_input = tf.placeholder(tf.int64, [None, ])
    #-------------------------------Test-----------------------------------------
    if not os.path.exists('./trans/test.tfrecords'):
        generate_tfrecord(x_test, y_test, './trans/', 'test.tfrecords')
    dataset_test = tf.data.TFRecordDataset('./trans/test.tfrecords')
    dataset_test = dataset_test.map(parse_test)
    dataset_test = dataset_test.shuffle(buffer_size=10000)
    dataset_test = dataset_test.batch(128)
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next() 

    prob_test = densenet201(x_input, is_training=False, reuse=False, kernel_initializer=None)

    
    # prob_test = tf.layers.dense(prob_test, 100, reuse=True, name='before_softmax')
    logit_softmax_test = tf.nn.softmax(prob_test)
    acc_test = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logit_softmax_test, 1), y_input), tf.float32))

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    saver = tf.train.Saver(var_list=var_list)
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True



    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt)
        sess.run(iterator_test.initializer)
        avg_acc = []
        while True:
            try:
                batch_test, label_test = sess.run(next_element_test)
                acc_test_val = sess.run(acc_test, feed_dict={x_input: batch_test, y_input: label_test})
                avg_acc.append(acc_test_val)
            except tf.errors.OutOfRangeError:
                print('end test ', np.sum(avg_acc)/len(y_test))
                break




train(batch_size=64, epoch=200, network = densenet201, opt = 'momentum', train_path = '/home/dylan_dwork//PycharmProjects/Cifar-100/cifar-100-python/train', test_path = '/home/dylan_dwork/PycharmProjects/Cifar-100/cifar-100-python/test')

test(network=densenet201, test_path='/home/dylan_dwork/PycharmProjects/Cifar-100/cifar-100-python/test',ckpt='params/resnet18/Speaker_vox_iter_58000.ckpt')
