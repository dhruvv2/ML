# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP5. You should only modify code
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> h ->  out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        lrate = 0.01
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size
        # print(in_size)
        # print(out_size)
        self.nnet = nn.Sequential(nn.Linear(self.in_size, 31), nn.ReLU(), nn.Linear(31, self.out_size))
        self.optimizer = optim.SGD(self.nnet.parameters(), lrate)
        
        #raise NotImplementedError("You need to write this part!")
    

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #raise NotImplementedError("You need to write this part!")
        return self.nnet(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        #raise NotImplementedError("You need to write this part!")
        
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        self.loss_fn(yhat,y).backward()
        self.optimizer.step()
        return self.loss_fn(yhat,y).item()
        
        #return 0.0



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    #raise NotImplementedError("You need to write this part!")
    lrate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    in_size = train_set.shape[1]
    out_size = 4
    net = NeuralNet(lrate, loss_fn, in_size, out_size)
    yhats = np.ones(len(dev_set))
    
    losses = []
    
    train_set_stand = (train_set - train_set.mean()) / train_set.std()
    dev_set_stand = (dev_set - dev_set.mean()) / dev_set.std()
    
    # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    dataset = get_dataset_from_arrays(train_set_stand, train_labels)
    params = {'batch_size': batch_size, 'shuffle' : False}
    dataset_generator = torch.utils.data.DataLoader(dataset, **params)
    for epoch in range(epochs):
        for dicts in dataset_generator:
            labels = dicts["labels"]
            features = dicts["features"]
            step = net.step(features, labels)
            losses.append(step)
            
    forward = net.forward(dev_set_stand)
    for i in range(len(forward)):
        yhats[i] = torch.argmax(forward[i])
        
    yhats = yhats.astype(np.int)
    
    return losses,yhats,net
