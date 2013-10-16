#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Description: This python script integrate all util functions of machine learning
# Author: Feng Liang
# Email: liangfeng1987@gmail.com
# Date: 2013-10-13 10:15:11
# Last Modified: 2013-10-16 16:39:29

from numpy import zeros, random, tile, shape, array


def load_data(file_name, delimiter, at_last_column=True):
    '''
    Load data matrix and label vector from the given file,
    by default, label is placed at last column
    '''
    # open the data file, calculate the number of instance and the number of features,
    # and build a matrix with size of # of instance by # of feature.
    in_file = open(file_name, 'r')
    file_lines = in_file.readlines()
    instance_num = len(file_lines)
    feature_num = len(file_lines[0].strip().split(delimiter)) - 1
    data_matrix = zeros((instance_num, feature_num))
    label_vector = []

    index = 0
    for line in file_lines:
        line = line.strip()
        splited_fields = line.split(delimiter)
        # assert only for debug
        assert len(splited_fields) == feature_num + 1
        if at_last_column:
            data_matrix[index, :] = splited_fields[0:feature_num]
            label_vector.append(splited_fields[-1])
        else:
            data_matrix[index, :] = splited_fields[1:feature_num+1]
            label_vector.append(splited_fields[0])
        index += 1

    return data_matrix, array(label_vector)


def auto_normalize(dataset):
    '''
    Normalize the dataset automatically according to new_val = (old_value-min) / (max-min)
    '''
    # find the maximum and minimum value of each feature
    min_features = dataset.min(0)
    max_features = dataset.max(0)
    range_features = max_features - min_features

    normalized_dataset = zeros(shape(dataset))
    data_size = dataset.shape[0]
    ex_min_features = tile(min_features, (data_size, 1))
    ex_range_features = tile(range_features, (data_size, 1))

    normalized_dataset = (dataset - ex_min_features) / ex_range_features

    return normalized_dataset, min_features, range_features


def split_dataset(dataset, label, fold=5):
    '''
    Split dataset into $fold parts equally, the default fold is 5
    '''
    if fold < 2 or fold > dataset.shape[0]:
        raise ValueError(
            "the parameter fold should greater than 2 and less than"
            " the number of samples in dataset")

    if not len(dataset) == len(label):
        raise ValueError(
            "the dataset matrix and label array should have the same "
            "leading dimension")

    # random shuffle dataset
    perm = random.permutation(len(label))
    random_dataset = dataset[perm]
    random_label = label[perm]

    instance_num = dataset.shape[0]
    sample_num = int(instance_num/fold)

    dataset_ret = []
    label_ret = []
    start = 0
    for i in range(fold):
        if i < fold - 1:
            dataset_ret.append(random_dataset[start:start+sample_num, :])
            label_ret.append(random_label[start:start+sample_num])
        else:
            dataset_ret.append(random_dataset[start:, :])
            label_ret.append(random_label[start:])

        start += sample_num

    return dataset_ret, label_ret


def cal_accuracy(predicted_labels, ground_truth_labels):
    '''
    Calculate error rate for classification problem
    '''
    # assert two label vectors have the same length
    if not len(predicted_labels) == len(ground_truth_labels):
        raise ValueError(
            "Two label arrays should have the same length")

    size = len(predicted_labels)
    correct_num = 0

    for index in range(size):
        if predicted_labels[index] == ground_truth_labels[index]:
            correct_num += 1

    return float(correct_num) / float(size)
