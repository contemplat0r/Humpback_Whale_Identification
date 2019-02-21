# -*- coding: utf-8 -*-

import random

import torch

def select_objects(indexes_list, objects_names):
    return tuple(objects_names[i] for i in indexes_list)

def select_random_indexses_subset(size, subset_size):
    return random.sample(tuple(range(size)), subset_size)

def random_objects_select(objects_names, subset_size):
    objects_names_len = len(objects_names)
    indexes = select_random_indexses_subset(objects_names_len, subset_size)
    return select_objects(indexes, objects_names)

def select_offset_indexses_subset(size, subset_size, offset):
    return tuple(range(size))[offset:offset + subset_size]

def offset_objects_select(objects_names, subset_size, offset):
    objects_names_len = len(objects_names)
    indexes = select_offset_indexses_subset(objects_names_len, subset_size, offset)
    return select_objects(indexes, objects_names)

def save_model(model, full=True, name='model'):
    if not full:
        torch.save(model.state_dict(), '{}_params.pkl'.format(name))
    else:
        torch.save(model, '{}.pkl'.format(name))
    return


def crossvalidation(model, dataset, targets, dataset_size, partition_func, train_func, validation_func, folds_num, model_name='model'):
    indices_per_fold = dataset_size / folds_num

    validation_metric_values = []

    for fold_num in range(folds_num - 1):
        if fold_num == 0:
            validation_dataset_part = partition_func(fold_num * indices_per_fold, (fold_num + 1) * indices_per_fold)
            #train_dataset_part = partition_func(0, fold_num * indices_per_fold) + partition_func((fold_num + 1) * indices_per_fold, dataset_size)
            train_dataset_part = partition_func((fold_num + 1) * indices_per_fold, dataset_size)
        elif fold_num == (folds_num - 1):
            #validation_dataset_part = partition_func(fold_num * indices_per_fold, (fold_num + 1) * indices_per_fold)
            validation_dataset_part = partition_func(fold_num * indices_per_fold, dataset_size)
        else:
            validation_dataset_part = partition_func(fold_num * indices_per_fold, (fold_num + 1) * indices_per_fold)
            train_dataset_part = partition_func(0, fold_num * indices_per_fold) + partition_func((fold_num + 1) * indices_per_fold, dataset_size)
        train_func(model, train_dataset_part)
        validation_metric_values.append(validation_func(model, validation_dataset_part))

    return validation_metric_values


#def simple_offset_crossvalidation(model, dataset, targets, dataset_size, partition_func, train_func, validation_func, metric_calculator, folds_num):
def simple_offset_crossvalidation(model, dataset, targets, dataset_size, train_func, validation_func, metric_calculator, folds_num, model_name='model'):
    indices_per_fold = dataset_size / folds_num

    train_losses_values = []
    validation_losses_values = []
    validation_metric_values = []
    train_metrics_values = []
    validation_predicted_targets_values = []
    train_predicted_targets_values = []


    for fold_num in range(folds_num - 1):
        if fold_num == 0:
            validation_dataset_part = dataset[fold_num * indices_per_fold:(fold_num + 1) * indices_per_fold]
            train_dataset_part = dataset[(fold_num + 1) * indices_per_fold:dataset_size]
            validation_targets_part = targets[fold_num * indices_per_fold:(fold_num + 1) * indices_per_fold]
            train_targets_part = targets[(fold_num + 1) * indices_per_fold:dataset_size]
        elif fold_num == (folds_num - 1):
            validation_dataset_part = dataset[fold_num * indices_per_fold:dataset_size]
            train_dataset_part = dataset[0:(fold_num - 1) * indices_per_fold]
            validation_targets_part = targets[fold_num * indices_per_fold:dataset_size]
            train_targets_part = targets[0:(fold_num - 1) * indices_per_fold]
        else:
            validation_dataset_part = dataset[fold_num * indices_per_fold:(fold_num + 1) * indices_per_fold]
            train_dataset_part = dataset[0:fold_num * indices_per_fold] + dataset[(fold_num + 1) * indices_per_fold:dataset_size]
            validation_targets_part = targets[fold_num * indices_per_fold:(fold_num + 1) * indices_per_fold]
            train_targets_part = targets[0:fold_num * indices_per_fold] + targets[(fold_num + 1) * indices_per_fold:dataset_size]
        # Output format for train_func, and validation_funct must be in concordance (match?, coincide?) with input data format
        # of metric_calulcator. By default - (predicted) values of targets. But in dependence of train/validation func format can be,
        # for example, bollean vector with results of comparison of predicted and true target values. Or (in case of this 
        # competition) not standart. Code must be extended to captre this situation? In this cases true target (train/validation)
        # values and metric_calculator must be give (as parameters) to train_func/validation_func and in code must be reailized
        # corresponding "if" statement (to select code with "plain" train/validation functions that receive, as parameters,
        # model and train/val dataset part and code with "complex" train/val functions that receive also true targets values,
        # and metric_calculator. Parameters of this function (simple_offset_crossvalidation) must be contains boolean value
        # for switching between described options.
        model, train_losses, train_predicted_targets = train_func(model, train_dataset_part)
        save_model(model, model_name='{}_{}'.format(model_name, str(fold_num)))
        train_losses_values.append(train_losses)
        train_predicted_targets_values.append(train_predicted_targets)

        validation_losses, validation_predicted_targets = validation_func(model, validation_dataset_part)
        validation_losses_values.append(validation_losses)
        validation_predicted_targets_values.append(validation_predicted_targets)

        train_metrics_values.append(metric_calculator(train_predicted_targets, train_targets_part))
        validation_metric_values.append(metric_calculator(validation_predicted_targets, validation_targets_part))

    return (
            train_metrics_values,
            validation_metric_values,
            train_losses_values,
            validation_losses_values
            train_targets_part,
            train_predicted_targets,
            validation_targets_part,
            validation_predicted_targets
        )


