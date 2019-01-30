# -*- coding: utf-8 -*-

def crossvalidation(model, dataset, dataset_size, folds_num, get_datset_part_func, train_func, validation_func):
    indices_per_fold = dataset_size / folds_num

    validation_metric_values = []

    for fold_num in range(folds_num - 1):
        if fold_num == 0:
            validation_dataset_part = get_datset_part_func(fold_num * indices_per_fold, (fold_num + 1) * indices_per_fold)
            #train_dataset_part = get_datset_part_func(0, fold_num * indices_per_fold) + get_datset_part_func((fold_num + 1) * indices_per_fold, dataset_size)
            train_dataset_part = get_datset_part_func((fold_num + 1) * indices_per_fold, dataset_size)
        elif fold_num == (folds_num - 1):
            #validation_dataset_part = get_datset_part_func(fold_num * indices_per_fold, (fold_num + 1) * indices_per_fold)
            validation_dataset_part = get_datset_part_func(fold_num * indices_per_fold, dataset_size)
        else:
            validation_dataset_part = get_datset_part_func(fold_num * indices_per_fold, (fold_num + 1) * indices_per_fold)
            train_dataset_part = get_datset_part_func(0, fold_num * indices_per_fold) + get_datset_part_func((fold_num + 1) * indices_per_fold, dataset_size)
        train_func(model, train_dataset_part)
        validation_metric_values.append(validation_func(model, validation_dataset_part))


    return validation_metric_values
