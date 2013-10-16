#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Description: KNN classification algorithm and application
# Author: Feng Liang
# Email: liangfeng1987@gmail.com
# Date: 2013-10-11 11:40:42
# Last Modified: 2013-10-16 16:54:59

from operator import itemgetter

from numpy import tile, vstack, hstack

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


def train_and_test(train_file_name, test_file_name, delimiter, k, is_auto_normalized=True):
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


def run_optdigit_cv(train_file_name, test_file_name, delimiter, max_k, is_auto_normalized=True):
    '''
    Automatically test KNN algorithm using optdigits dataset with cross validation
    '''
    # load train data and test data, respectively
    train_data_matrix, train_labels = load_data(train_file_name, delimiter)
    test_data_matrix, test_labels = load_data(test_file_name, delimiter)

    # if dataset is required to be normalized, then use auto_normalize()
    if is_auto_normalized:
        train_data_matrix, min_features, range_features = auto_normalize(train_data_matrix)

    # split train dataset into train dataset and validation dataset
    # by default, using 5-fold cross validtion
    fold = 5
    splited_train_dataset, splited_labels = split_dataset(train_data_matrix, train_labels, fold)

    # record the performances with different k
    best_k = -1
    best_performance = 0.0

    for k in range(1, max_k):
        current_performance = 0.0
        for i in range(fold):
            valid_data_matrix = splited_train_dataset[i]
            valid_labels = splited_labels[i]
            valid_data_size = valid_data_matrix.shape[0]

            # copy the splited_train_dataset and delete the validation dataset
            # to construct new training set
            train_data_list = list(splited_train_dataset)
            del train_data_list[i]
            train_labels_list = list(splited_labels)
            del train_labels_list[i]

            train_data_matrix = vstack(train_data_list)
            train_labels = hstack(train_labels_list)

            predicted_labels = []
            # training at train dataset and test at validation dataset
            for index in range(valid_data_size):
                valid_vect = valid_data_matrix[index, :]
                if is_auto_normalized:
                    valid_vect = (valid_vect - min_features) / range_features

                label = classify(valid_vect, train_data_matrix, train_labels, k)
                predicted_labels.append(label)

            current_performance += cal_accuracy(predicted_labels, valid_labels)

        current_performance /= fold

        print "k=%i: average performance is %f" % (k, current_performance)
        # update the best performance
        if current_performance > best_performance:
            best_performance = current_performance
            best_k = k

    print "Cross Validation over, re-training begins with best k of %i..." % best_k
    all_accurate = train_and_test(
        train_file_name, test_file_name, delimiter, best_k, is_auto_normalized)
    print "Final test result with k = %i, accuracy is %f" % (best_k, all_accurate)


def run_test_manually(train_file_name, test_file_name, delimiter, max_k, is_auto_normalized=True):
    for i in range(1, max_k):
        error_rate = train_and_test(
            train_file_name, test_file_name, delimiter, i, is_auto_normalized)
        print "k = %i : %f" % (i, 100*error_rate)


if __name__ == '__main__':
    delimiter = ','
    max_k = 12
    train_file_name = 'dataset/optdigits.tra'
    test_file_name = 'dataset/optdigits.tes'
    is_auto_normalized = False

    #run_test_manually(train_file_name, test_file_name, delimiter, max_k, is_auto_normalized)
    run_optdigit_cv(train_file_name, test_file_name, delimiter, max_k, is_auto_normalized)
