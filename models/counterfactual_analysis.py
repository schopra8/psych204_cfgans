# --------------------------------------------------------------------------------------------------
# Author: Sahil Chopra
# Psych 204, Fall 2017
# Goal: Given a Generator Model, load its weights and sample its worlds. 
#       When we find a sample that seems to produce a realistic world,
#       continue to perturb this vector for a 100 iterations according a perturbation format,
#       note the distribution of worlds that are produced.
# --------------------------------------------------------------------------------------------------

import argparse
import os
import torch
from blicket_model import Generator, on_gpu, produce_samples

if __name__ == '__main__':
    # Generator Definiton
    g_input_size = 5
    g_hidden_size = 10
    g_output_size = 4
    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    if on_gpu():
        G = G.cuda()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to Generator Model Weights')
    args = parser.parse_args()

    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            G.load_state_dict(checkpoint['state_dict'])
            print("=> successfully loaded checkpoint")
        else:
            print("=> no checkpoint found at '{}'".format(args.model))

    samples = produce_samples(G, g_input_size)
    print samples


def perturbation_one():
    # z_i' = flip(s) ? z_i : prior(z_i)  
    pass 

def perturbation_two():
    # z_i' = N(z_i, sigma)
    pass

def perturbation_three():
    # z_i' = flip(s) ? z_i : N(z_i, sigma)
    pass
