#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Description: KNN classification algorithm and application
# Author: Feng Liang
# Email: liangfeng1987@gmail.com
# Date: 2013-10-11 11:40:42
# Last Modified: 2013-10-15 15:52:54

from operator import itemgetter

from numpy import tile

from mlAction.ml_util import auto_normalize, load_data, split_dataset, cal_accuracy

def classify(in_vect, train_vects, train_labels, k):
    '''
    KNN algorithm framework,
    '''
    # assert input vector and train vectors have the same dimention
    assert in_vect.shape[0] == train_vects.shape[1]
    # get the number of training instances
    train_num = train_vects.shape[0]

    # expand test vector into m vectors
    diff_matrix = tile(in_vect, (train_num, 1))

    # calculate distances
    diff_matrix = diff_matrix - train_vects
    square_diff_matrix = diff_matrix**2
    # sum all square of differences of attributes in column (specify axis = 1)
    square_distance = square_diff_matrix.sum(axis=1)
    distance_vector = square_distance**0.5

    # calculate the sorted indicies
    sorted_dist_indicies = distance_vector.argsort()

    # count the label numbers within k minimum distances
    count_labels = {}
    for i in range(k):
        label = train_labels[sorted_dist_indicies[i]]
        count_labels[label] = count_labels.get(label, 0) + 1

    # sort the labels counter and return the label that most training instances hold
    sorted_list = sorted(count_labels.iteritems(), key=itemgetter(1), reverse=True)

    return sorted_list[0][0]


def opt_digit_test(train_file_name, test_file_name, delimiter, k, is_auto_normalized=True):
    '''
    Test KNN algorithm using optdigits dataset
    '''
    # load train data and test data, respectively
    train_data_matrix, train_labels = load_data(train_file_name, delimiter)
    test_data_matrix, test_labels = load_data(test_file_name, delimiter)

    # if dataset is required to be normalized, then use auto_normalize()
    if is_auto_normalized:
        train_data_matrix, min_features, range_features = auto_normalize(train_data_matrix)

    predicted_labels = []
    test_data_size = test_data_matrix.shape[0]
    for index in range(test_data_size):
        test_vect = test_data_matrix[index, :]
        if is_auto_normalized:
            test_vect = (test_vect - min_features) / range_features
        predicted_label = classify(test_vect, train_data_matrix, train_labels, k)
        predicted_labels.append(predicted_label)

    return cal_accuracy(predicted_labels, test_labels)


if __name__ == '__main__':
    delimiter = ','
    max_k = 12
    train_file_name = 'dataset/optdigits.tra'
    test_file_name = 'dataset/optdigits.tes'
    is_auto_normalized = False

    for i in range(1, max_k):
        error_rate = opt_digit_test(
            train_file_name, test_file_name, delimiter, i, is_auto_normalized)
        print "k = %i : %f" % (i, 100*error_rate)
