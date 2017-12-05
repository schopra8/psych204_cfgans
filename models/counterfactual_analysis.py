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
import random
import numpy.random.normal as normal
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
    parser.add_argument('--model', help='Path to Generator Model Weights', default='./checkpts/g_1000000_0.3_checkpoint.pth.tar')
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
    counterfactualize(G, z, samples)

def check_correct(result, cut_off, stats):
    if result[0] >= cut_off and result[2] >= cut_off and result[3] >= 0.90:
        return True, 'D_ON'
    elif result[0] <= cut_off and result[3] < 0.90:
        if result[2] <= cut_off:
            return True, 'D_OFF_A_OFF_C_OFF'
        else:
            return True, 'D_OFF_A_OFF_C_ON'
    elif result[2] <= cut_off and result[3] < 0.90:
        return True, 'D_OFF_A_ON_C_OFF'
    else:
        return False, None

def flip(s):
    return random.uniform(0, 1) >= s

def prior():
    return normal()

def counterfactualize(G, Z, samples):
    for s in samples:
        valid_sample, state = check_correct(s)
        if valid_sample:
            print 'state: {}'.format(state)
            print 'original: {}'.format(Z[i, :])
            print perturbation_one(Z[i, :])
            break
    pass


def perturbation_one(z, s=0.5):
    z_i = z.clone()
    for j in z.size()[1]
        if not flip(s):
            z_i[j] = normal()
    return z_i

def perturbation_two():
    z_i = z.clone()
    # z_i = normal
    # z_i' = N(z_i, sigma)
    pass

def perturbation_three():
    # z_i' = flip(s) ? z_i : N(z_i, sigma)
    pass
