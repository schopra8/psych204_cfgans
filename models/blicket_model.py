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
import matplotlib.pyplot as plt

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
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')


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
    d_optimizer = optim.Adam(D.parameters(), lr=lr)
    g_optimizer = optim.Adam(G.parameters(), lr=lr)

    d_losses = []
    g_losses = []    

    for epoch in range(num_epochs):
        print "Epoch {} of {}".format(epoch + 1, num_epochs)
        d_errors = []
        g_errors = []

        for i, minibatch in enumerate(data):
            minibatch = Variable(minibatch)
            num_samples = minibatch.size()[0]

            # Train Discriminator on Real Data
            D.zero_grad()
            d_real_preds = D(minibatch)
            d_real_error = criterion(d_real_preds, Variable(torch.ones(num_samples)).type(torch.LongTensor))
            d_real_error.backward()

            # Train Discriminator on Fake Data
            sample = torch.FloatTensor(num_samples, g_input_size)
            sample.normal_()
            gaussian_sample = Variable(sample)
            d_fake_input = G(gaussian_sample).detach()
            d_fake_preds = D(d_fake_input)
            d_fake_error = criterion(d_fake_preds, Variable(torch.zeros(num_samples)).type(torch.LongTensor))
            d_fake_error.backward()
            d_errors.append(d_fake_error.data[0])
            d_errors.append(d_real_error.data[0])
            d_optimizer.step() # Update Discriminator's params now that we've processed real and fake

            # Train Generator
            if epoch >= 5 and i % 2 == 0:
                G.zero_grad()
                gaussian_sample = Variable(sample)
                d_fake_input = G(gaussian_sample)
                d_fake_preds = D(d_fake_input)
                g_error = criterion(d_fake_preds, Variable(torch.ones(num_samples).type(torch.LongTensor)))
                g_error.backward()
                g_optimizer.step() # Update Generator's parameters
                g_errors.append(g_error.data[0])

        d_epoch_avg_loss = np.mean(d_errors)
        d_losses.append(d_epoch_avg_loss)

        if epoch >= 5:
            g_epoch_avg_loss = np.mean(g_errors)
            g_losses.append(g_epoch_avg_loss)
            print "Discriminator Loss: {} Generator Loss: {}".format(d_epoch_avg_loss, g_epoch_avg_loss)
        else:
            print "Discriminator Loss: {}".format(d_epoch_avg_loss)

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

    return D, G, g_input_size, d_losses, g_losses

def produce_samples(G, g_input_size):
    gaussian_sample = torch.FloatTensor(100, g_input_size)
    gaussian_sample.normal_()
    gaussian_sample = Variable(gaussian_sample, requires_grad=False)
    gen_outputs = G(gaussian_sample)
    return gen_outputs    

if __name__ == '__main__':
    blicket_data = parse_blicket_data('../gen_data/data/blicket_test_40_p.json')
    D, G, g_input_size, d_losses, g_losses = train(num_epochs=50, save_epochs=10, data=blicket_data)
    
    outputs = produce_samples(G, g_input_size)
    num_correct = 0
    num_incorrect = 0
    num_correct_with_large_b = 0
    num_correct_with_small_b = 0
    num_incorrect_with_small_b = 0
    num_incorrect_with_large_b = 0
    for o in outputs.split(1):
        if o.data[0] >= 0.4 and o.data[2] >= 0.4 and o.data[3] >= 0.80:
            num_correct += 1
            if o.data[1] <= 0.5:
                num_correct_with_small_b += 1
            else:
                num_correct_with_large_b += 1
        else:
            num_incorrect += 1
            if o.data[1] <= 0.5:
                num_incorrect_with_small_b += 1
            else:
                num_incorrect_with_large_b += 1            

    print '-'*80
    print "Num Correct: {}".format(num_correct)
    print "Num Incorrect: {}".format(num_incorrect)
    print "Num Correct with Small B: {}".format(num_correct_with_small_b)
    print "Num Correct with Large B: {}".format(num_correct_with_large_b)
    print "Num Incorrect with Small B: {}".format(num_incorrect_with_small_b)
    print "Num Incorrect wtih Large B: {}".format(num_incorrect_with_large_b)

    d_plot = plt.plot(range(1, 51), d_losses, 'r--', label="Discriminator")
    g_plot = plt.plot(range(1, 51), g_losses, 'b++', label="Generator")
    plt.title('Loss vs. Time')
    plt.legend(handles=[d_plot, g_plot])
    plt.show()




