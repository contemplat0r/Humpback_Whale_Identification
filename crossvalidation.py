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

def prepare_dataset_loaders(dataset, valid_train_ratio=0.6):
    dataset_size = len(dataset)
    print("dataset_size: ", dataset_size)

    validation_subset_size = int(dataset_size * (1 - valid_train_ratio))
    print("validation_subset_size: ", validation_subset_size)

    indices = list(range(dataset_size))
    validation_indices = np.random.choice(indices, size=validation_subset_size, replace=False)
    train_indices = list(set(indices) - set(validation_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    
    dataset_sizes = {
            'train': len(train_indices),
            'validation': len(validation_indices)
        }

    #train_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, sampler=train_sampler, pin_memory=True)
    train_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, sampler=train_sampler)
    #validation_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, sampler=validation_sampler, pin_memory=True)
    validation_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, sampler=validation_sampler)
    loaders = {
            'train': train_loader,
            'validation': validation_loader
        }

    return loaders, dataset_sizes

def unfold_batch(batch):
    return batch['image'], batch['label']

def one_epoch_train(model, data_loader, criterion, optimizer):
    
    accuracy = 0.0
    total_loss = 0.0
    correct_predicted_total = 0.0
    
    for i, data_batch in enumerate(data_loader, 0):
        
        inputs, labels = unfold_batch(data_batch)
        if inputs.size()[0] == BATCH_SIZE:
        
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
        
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item() * inputs.size(0)
        
            predicted = outputs > 0
            labels = labels.data.byte()
            sum_of_correct_predicted = torch.sum((predicted[0] == labels))
            
            item = sum_of_correct_predicted.item()
            correct_predicted_total += item
        
    accuracy = correct_predicted_total
    
    #epoch_train_accuracy = correct_predicted_total / train_dataset_size
    return (total_loss, accuracy)

def one_epoch_train(model, data_loader, criterion, optimizer):
    
    accuracy = 0.0
    total_loss = 0.0
    correct_predicted_total = 0.0
    
    for i, data_batch in enumerate(data_loader, 0):
        
        inputs, labels = unfold_batch(data_batch)
        if inputs.size()[0] == BATCH_SIZE:
        
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
        
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item() * inputs.size(0)
        
            predicted = outputs > 0
            labels = labels.data.byte()
            sum_of_correct_predicted = torch.sum((predicted[0] == labels))
            
            item = sum_of_correct_predicted.item()
            correct_predicted_total += item
        
    #accuracy = correct_predicted_total
    
    #epoch_train_accuracy = correct_predicted_total / train_dataset_size
    return (total_loss, predicted, labels)


def one_epoch_validate(model, data_loader, criterion):
    model.eval()
    with torch.no_grad():
        
        correct_predicted_total = 0.0
        total_loss = 0.0

        for data_batch in data_loader:
            inputs, labels = unfold_batch(data_batch)
            if inputs.size()[0] == BATCH_SIZE:

                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                #labels = labels.to(device, dtype=torch.long)
                outputs = model(inputs)
            
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
            
                predicted = outputs > 0
            
                labels = labels.data.byte()
                sum_of_correct_predicted = torch.sum((predicted[0] == labels))
                item = sum_of_correct_predicted.item()

                correct_predicted_total += item

        accuracy = correct_predicted_total        

    return (total_loss, accuracy)

def one_epoch_validate(model, data_loader, criterion):
    model.eval()
    with torch.no_grad():
        
        correct_predicted_total = 0.0
        total_loss = 0.0

        for data_batch in data_loader:
            inputs, labels = unfold_batch(data_batch)
            if inputs.size()[0] == BATCH_SIZE:

                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                #labels = labels.to(device, dtype=torch.long)
                outputs = model(inputs)
            
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
            
                predicted = outputs > 0
            
                labels = labels.data.byte()
                sum_of_correct_predicted = torch.sum((predicted[0] == labels))
                item = sum_of_correct_predicted.item()

                correct_predicted_total += item

        #accuracy = correct_predicted_total        

    return (total_loss, predicted, labels)


def train_model(num_of_epoch, model, dataset_loaders, dataset_sizes, criterion, optimizer):
    torch.cuda.empty_cache()
    since = time.time()
    
    train_loader = dataset_loaders['train']
    validation_loader = dataset_loaders['validation']
    train_dataset_size = dataset_sizes['train']
    validation_dataset_size = dataset_sizes['validation']
    
    best_model_accuracy = 0.0
    best_model_weights = model.state_dict()
    
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    
    for epoch in range(num_of_epoch):
        
        train_loss, train_accuracy = one_epoch_train(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss / train_dataset_size)
        train_accuracies.append(train_accuracy / train_dataset_size)
        
        validation_loss, validation_accuracy = one_epoch_validate(model, validation_loader, criterion)
        validation_losses.append(validation_loss / validation_dataset_size)
        validation_accuracies.append(validation_accuracy / validation_dataset_size)
        
        if validation_accuracy > best_model_accuracy:
            best_model_accuracy = validation_accuracy
            best_model_weights = model.state_dict()
        
        print("Epoch {}: train loss {}, train accuracy"
          " {}, validation loss {}, validation accuracy {}".format(
              epoch + 1,
              train_loss / train_dataset_size,
              train_accuracy / train_dataset_size,
              validation_loss / validation_dataset_size,
              validation_accuracy / validation_dataset_size
            )
        )
    print("Finished Training")
    time_elapsed = time.time() - since
    print(
            'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best model accuracy: ", best_model_accuracy / validation_dataset_size)
    model.load_state_dict(best_model_weights)
    return train_losses, validation_losses, train_accuracies, validation_accuracies

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

def crossvalidation(
        model,
        dataset,
        train_func,
        validation_func,
        metric_calculator,
        criterion,
        optimizer,
        folds_num,
        epoch_num,
        model_name='model'
    ):

    torch.cuda.empty_cache()
    since = time.time()

    best_model_metric = 0.0
    best_model_weights = model.state_dict()

    indices_per_fold = dataset_size / folds_num

    train_valid_ratio = (folds_num - 1.0) / folds_num

    train_losses_values = []
    validation_losses_values = []

    validation_predicted_targets_values = []
    train_predicted_targets_values = []

    validation_metric_values = []
    train_metrics_values = []


    for fold_num in range(folds_num - 1):

        dataset_loaders, dataset_sizes = prepare_dataset_loaders(dataset, train_valid_ratio)
        train_dataset_loader = dataset_loaders['train']
        validation_dataset_loader = dataset_loaders['validation']

        for epoch in range(num_of_epoch):
            eposh_train_losses_values = []
            eposh_validation_losses_values = []
            eposh_validation_metric_values = []
            eposh_train_metrics_values = []
            eposh_validation_predicted_targets_values = []
            eposh_train_predicted_targets_values = []

            # Output format for train_func, and validation_funct must be in concordance (match?, coincide?) with input data format
            # of metric_calulcator. By default - (predicted) values of targets. But in dependence of train/validation func format can be,
            # for example, bollean vector with results of comparison of predicted and true target values. Or (in case of this 
            # competition) not standart. Code must be extended to captre this situation? In this cases true target (train/validation)
            # values and metric_calculator must be give (as parameters) to train_func/validation_func and in code must be reailized
            # corresponding "if" statement (to select code with "plain" train/validation functions that receive, as parameters,
            # model and train/val dataset part and code with "complex" train/val functions that receive also true targets values,
            # and metric_calculator. Parameters of this function (simple_offset_crossvalidation) must be contains boolean value
            # for switching between described options.
            model, train_losses, train_predicted_targets, train_true_targets = train_func(
                    model,
                    train_dataset_loader,
                    criterion,
                    optimizer
                )
            save_model(model, model_name='{}_{}'.format(model_name, str(fold_num)))

            eposh_train_losses_values.append(train_losses)
            eposh_train_predicted_targets_values.append(train_predicted_targets)

            eposh_validation_losses, validation_predicted_targets, validation_true_targets = validation_func(
                    model,
                    validation_dataset_loader,
                    criterion
                )

            eposh_validation_losses_values.append(validation_losses)
            eposh_validation_predicted_targets_values.append(validation_predicted_targets)

            eposh_train_metrics_values.append(metric_calculator(train_predicted_targets, train_targets_part))
            eposh_validation_metric_values.append(metric_calculator(validation_predicted_targets, validation_targets_part))

            if eposh_validation_metric_values[-1] > best_model_metric:
                best_model_metric = eposh_validation_metric_values[-1]
                best_model_weights = model.state_dict()

        train_losses_values.append(eposh_train_losses_values)
        validation_losses_values.append(eposh_validation_losses_values)

        train_predicted_targets_values.append(eposh_train_predicted_targets_values)
        validation_predicted_targets_values.append(eposh_validation_predicted_targets_values)

        train_metrics_values.append(eposh_train_metrics_values)

        validation_metric_values.append(eposh_validation_metric_values)

    print("Finished Training")
    time_elapsed = time.time() - since
    print(
            'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    model.load_state_dict(best_model_weights)

    return (
            train_losses_values,
            validation_losses_values,

            train_predicted_targets_values,
            validation_predicted_targets_values,

            train_metrics_values,
            validation_metric_values
        )

