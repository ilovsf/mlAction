#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Description: This python script integrate all util functions of machine learning
# Author: Feng Liang
# Email: liangfeng1987@gmail.com
# Date: 2013-10-13 10:15:11
# Last Modified: 2013-10-13 11:37:57

from numpy import zeros


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

    return data_matrix, label_vector


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

    normalized_dataset = (normalized_dataset - ex_min_features) / ex_range_features

    return normalized_dataset, min_features, range_features
