# --------------------------------------------------------------------------------------------------
# Author: Sahil Chopra
# Psych 204, Fall 2017
# Goal: Simple GAN of MLPs to model blicket test data.
# --------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import json
import shutil
from random import seed, shuffle

# Set random seed
seed(3)

def parse_blicket_data(fn):
    """
    fn: filename/filepath to blickets data
    """
    with open(fn) as blicket_data_file:
        data = json.load(blicket_data_file)
        vectorized_data = [row.values() for row in data]
        tensor_data = torch.FloatTensor(vectorized_data)
        blicket_data = torch.utils.data.DataLoader(
            tensor_data,
            batch_size=100,
            shuffle=True,
            num_workers=4
        )
        return blicket_data

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer1_leaky_relu = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer2_leaky_relu = nn.LeakyReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        layer1_output = self.layer1_leaky_relu(self.layer1(x))
        layer2_output = self.layer2_leaky_relu(self.layer2(layer1_output))
        return self.layer3(layer2_output)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer1_leaky_relu = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer2_leaky_relu = nn.LeakyReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        layer1_output = self.layer1_leaky_relu(self.layer1(x))
        layer2_output = self.layer2_leaky_relu(self.layer2(layer1_output))
        return self.layer3(layer2_output)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(num_epochs, save_epochs, data):
    # Generator parameters
    g_input_size = 5
    g_hidden_size = 10
    g_output_size = 4
    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)

    # Discriminator parameters
    d_input_size = 4
    d_hidden_size = 10
    d_output_size = 2
    D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)

    # Optimizers
    criterion = nn.CrossEntropyLoss()
    lr = 1e-4
    d_optimizer = optim.Adam(D.parameters())
    g_optimizer = optim.Adam(G.parameters())    

    for epoch in range(num_epochs):
        print "Epoch {} of {}".format(epoch + 1, num_epochs)
        for i, minibatch in enumerate(data):
            minibatch = Variable(minibatch, requires_grad=False)
            num_samples = minibatch.size()[0]

            # Train Discriminator on Real Data
            D.zero_grad()
            d_real_preds = D(minibatch)
            d_real_error = criterion(d_real_preds, Variable(torch.ones(num_samples)).type(torch.LongTensor))
            d_real_error.backward()

            # Train Discriminator on Fake Data
            gaussian_sample = torch.FloatTensor(num_samples, g_input_size)
            gaussian_sample.normal_()
            gaussian_sample = Variable(gaussian_sample, requires_grad=False)
            d_fake_input = G(gaussian_sample).detach()
            d_fake_preds = D(d_fake_input)
            d_fake_error = criterion(d_fake_preds, Variable(torch.zeros(num_samples)).type(torch.LongTensor))
            d_fake_error.backward()
            d_optimizer.step() # Update Discriminator's params now that we've processed real and fake

            # Train Generator
            if epoch >= 10:
                G.zero_grad()
                gaussian_sample = torch.FloatTensor(num_samples, g_input_size)
                gaussian_sample.normal_()
                gaussian_sample = Variable(gaussian_sample, requires_grad=False)
                gen_outputs = G(gaussian_sample)
                gen_outputs_decision = D(gen_outputs)
                g_error = criterion(gen_outputs_decision, Variable(torch.ones(num_samples).type(torch.LongTensor)))
                g_error.backward()
                g_optimizer.step() # Update Generator's parameters
        
            if i % 50 == 0:
                print "Iteration {} of {} ---- Discriminator Loss: {}".format(i, len(data), np.mean([d_fake_error.data[0], d_real_error.data[0]]))
                if epoch >= 10:
                    print "Iteration {} of {} ---- Generator Loss: {}".format(i, len(data), g_error.data[0])

        if epoch % save_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': D.state_dict(),
                'optimizer': d_optimizer.state_dict()
            }, True, filename='d_checkpoint.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': G.state_dict(),
                'optimizer': g_optimizer.state_dict()
            }, True, filename='g_checkpoint.pth.tar')

    return D, G, g_input_size

def produce_samples(G, g_input_size):
    gaussian_sample = torch.FloatTensor(10, g_input_size)
    gaussian_sample.normal_()
    gaussian_sample = Variable(gaussian_sample, requires_grad=False)
    gen_outputs = G(gaussian_sample)
    return gen_outputs    

if __name__ == '__main__':
    blicket_data = parse_blicket_data('../gen_data/data/blicket_test_40_p.json')
    D, G, g_input_size = train(num_epochs=100, save_epochs=10, data=blicket_data)
    print produce_samples(G, g_input_size)

