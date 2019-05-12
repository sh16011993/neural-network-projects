#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)
def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()

    # model = network2.Network([784, 20, 10])
    # # train the network using SGD
    # learn_rate = [0.0001, 0.001, 0.01]
    # batch_size = [128, 256, 384]
    # max_val_acc = 0
    # best_learn_rate = 0.001
    # best_batch_size = 128
    # best_metrics = ([]*4)
    # num_epochs = 100
    # for l in learn_rate:
    #     for b in batch_size:
    #         print('Trying out learn_rate = {} and batch_size = {}'.format(l,b))
    #         metrics = model.SGD(
    #             max_accuracy=max_val_acc,
    #             training_data=train_data,
    #             epochs=num_epochs,
    #             mini_batch_size=b,
    #             eta=l,
    #             lmbda = 0.0,
    #             evaluation_data=valid_data,
    #             monitor_evaluation_cost=True,
    #             monitor_evaluation_accuracy=True,
    #             monitor_training_cost=True,
    #             monitor_training_accuracy=True)
    #         for val_acc in metrics[1]:
    #             if val_acc > max_val_acc:
    #                 max_val_acc = val_acc
    #                 best_learn_rate = l
    #                 best_batch_size = b
    #                 best_metrics = metrics
    #
    #
    #
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.plot(list(range(num_epochs)),list(1-np.array(best_metrics[3])/len(train_data[0])), label='Training error')
    # ax.plot(list(range(num_epochs)),list(1-np.array(best_metrics[1])/len(valid_data[0])), label='Validation error')
    # ax.legend()
    # plt.ylabel('Error')
    # plt.xlabel('Epochs')
    # plt.title('Learning curve for training and validation data')
    # plt.savefig('./plot1.png', format = 'png')
    #
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.plot(list(range(num_epochs)),list(np.array(best_metrics[3])/len(train_data[0])), label='Training accuracy')
    # ax.plot(list(range(num_epochs)),list(np.array(best_metrics[1])/len(test_data[0])), label='Validation accuracy')
    # ax.legend()
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    # plt.title('Accuracy curve for Training and Validation data')
    # plt.savefig('./plot2.png', format = 'png')
    #
    # print('Best learn rate: {}'.format(best_learn_rate))
    # print('Best batch size: {}'.format(best_batch_size))

    model  = network2.load('../experiment/bestmodel.json')
    updated_train = [[],[]]

    encoded_valid_data=[]
    for idx in range(len(valid_data[1])):
        encoded = np.zeros((10,1))
        encoded[valid_data[1][idx]][0] = 1
        encoded_valid_data = encoded_valid_data + [encoded]

    updated_train[0] = train_data[0] + valid_data[0]
    updated_train[1] = train_data[1] + encoded_valid_data


    max_val_acc = 0
    num_epochs = 30
    best_learn_rate = 0.0001
    best_batch_size = 128
    best_metrics = model.SGD(
        max_accuracy=max_val_acc,
        training_data=updated_train,
        epochs=num_epochs,
        mini_batch_size=best_batch_size,
        eta=best_learn_rate,
        lmbda = 0.0,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    print('Training + Validation accuracy: {}'.format(np.amax(np.array(best_metrics[3]))/len(updated_train[0])))
    print('Testing accuracy: {}'.format(np.amax(np.array(best_metrics[1]))/len(test_data[0])))

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(list(range(num_epochs)),list(1-np.array(best_metrics[3])/len(updated_train[0])), label='(Training+Validation) error')
    ax.plot(list(range(num_epochs)),list(1-np.array(best_metrics[1])/len(test_data[0])), label='Test error')
    ax.legend()
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.title('Learning curve for (Training + Validation) and Test data')
    plt.savefig('./plot3.png', format = 'png')

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(list(range(num_epochs)),list(np.array(best_metrics[3])/len(updated_train[0])), label='(Training+Validation) accuracy')
    ax.plot(list(range(num_epochs)),list(np.array(best_metrics[1])/len(test_data[0])), label='Test accuracy')
    ax.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Accuracy curve for (Training + Validation) and Test data')
    plt.savefig('./plot4.png', format = 'png')



if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
