# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch.autograd import Variable

def get_data():
    

if __name__ == '__main__':

    x, y = get_data()

    w, b = get_weights()

    for i in range(500):
        y_pred = simple_network(x)
        loss = loss_fn(y, y_pred)
        if i % 50:
            print(loss)
        optimize(learning_rate)

