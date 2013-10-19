#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Description: Decision Tree classification algorithm and its application
# Author: Feng Liang
# Email: liangfeng1987@gmail.com
# Date: 2013-10-19 20:41:16
# Last Modified: 2013-10-19 22:28:03

from math import log


def create_test_data():
    '''
    Create dataset for testing
    '''
    dataset = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']

    return dataset, labels


def cal_entrophy(dataset):
    '''
    Calculate entrophy value for a dataset
    '''
    instance_num = len(dataset)
    label_dict = {}

    for instance in dataset:
        label = instance[-1]
        label_dict[label] = label_dict.get(label, 0) + 1

    entrophy = 0.0
    for label in label_dict:
        probability = float(label_dict[label]) / float(instance_num)
        entrophy -= probability * log(probability, 2)

    return entrophy


def split_dataset(dataset, axis, value):
    '''
    Split the dataset according to the value of the given axis (index)
    Example:
    [[2, 0, 'yes'], [1, 0, 'no'], [2, 1, 'yes']] split_dataset(dataset, 0, 2)
    ==> [[0, 'yes'], [1, 'yes']]
    Vertically, retain the instances whose value at axis is equal to the given value
    Horizonally, reduct the feature at the certain axis
    '''
    reducted_dataset = []
    for instance in dataset:
        if instance[axis] == value:
            new_feature = instance[:axis]
            new_feature.extend(instance[axis+1:])
            reducted_dataset.append(new_feature)

    return reducted_dataset


def choose_best_feature(dataset):
    '''
    Choose the best feature according to the information gain
    '''
    base_entrophy = cal_entrophy(dataset)
    feature_num = len(dataset[0]) - 1
    instance_num = len(dataset)

    best_feature_index = -1
    best_information_gain = 0.0
    for index in range(feature_num):
        feature_val_list = [ instance[index] for instance in dataset ]
        unique_val_list = set(feature_val_list)

        current_entrophy = 0.0
        for value in unique_val_list:
            splited_dataset = split_dataset(dataset, index, value)
            prob = float(len(splited_dataset)) / float(instance_num)
            current_entroph += prob * cal_entrophy(splited_dataset)

        if (base_entrophy - current_entrophy) > best_information_gain:
            best_information_gain = best_entrophy - current_entrophy
            best_feature_index = index

    return best_feature_index
