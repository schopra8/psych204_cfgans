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
import numpy as np
from blicket_model import Generator, on_gpu, produce_samples

def check_correct(result, cut_off):
    if result[0] <= cut_off and result[2] <= cut_off and result[3] < 0.90:
        return True, 'D_OFF_A_OFF_C_OFF'
    elif result[0] >= cut_off and result[2] >= cut_off and result[3] >= 0.90:
        return True, 'D_ON_A_ON_C_ON'
    elif result[0] <= cut_off and result[3] < 0.90:
        return True, 'D_OFF_A_OFF_C_ON'
    elif result[2] <= cut_off and result[3] < 0.90:
        return True, 'D_OFF_A_ON_C_OFF'
    else:
        return False, None

def flip(s):
    return random.uniform(0, 1) >= s

def prior():
    return np.random.normal()

def analysis(G, Z, samples):
    state_to_unary_conditions = {
        'D_ON_A_ON_C_ON': [(0, 0), (2, 0)], # Tuple (variable, condition)
        'D_OFF_A_OFF_C_OFF': [(0, 1), (2, 1)],
        'D_OFF_A_OFF_C_ON': [(0, 1), (2, 0)],
        'D_OFF_A_ON_C_OFF': [(0, 0), (2, 1)],
    }

    state_unary_condition_stats = {
        'D_ON_A_ON_C_ON': {
            (0, 0): 0,
            (2, 0): 0,
        }, 
        'D_OFF_A_OFF_C_OFF': {
            (0, 1): 0,
            (2, 1): 0,
        },
        'D_OFF_A_OFF_C_ON': {
            (0, 1): 0,
            (2, 0): 0,
        },
        'D_OFF_A_ON_C_OFF': {
            (0, 0): 0,
            (2, 1): 0,
        },
    }

    for i, s in enumerate(samples):
        valid_sample, state = check_correct(s.data, 0.3)
        if valid_sample and state == 'D_OFF_A_ON_C_OFF':
            for (var, cond) in state_to_unary_conditions[state]:
                print "Testing Condition:"
                percent_valid = conditionally_counterfactualize(G, Z[i, :], state, var, cond, perturbation=perturbation_one, num_observed=1000)
                state_unary_condition_stats['D_OFF_A_ON_C_OFF'][(var, cond)] = percent_valid
                print state_unary_condition_stats
            break
    print state_to_unary_conditions

def conditionally_counterfactualize(G, z_i, orig_state, var, cond, perturbation, num_observed=50):
    curr_observed = 0
    causal_link_retained = 0
    while curr_observed < num_observed:
        z_i_prime = perturbation(z_i)
        new_output = G(z_i_prime)
        valid, new_state = check_correct(new_output.data, 0.3)
        if (new_output.data[var] >= 0.3) == cond:
            curr_observed += 1
            
            print "Curr Observed: {}".format(curr_observed)
            print new_output
            print new_state
            if new_state is None:
                continue
            else:
                causal_link_retained += 1
    return (causal_link_retained * 1.0) / num_observed * 100

def perturbation_one(z, s=0.5):
    z_i = z.clone()
    for j in range(z.size()[0]):
        if not flip(0.5):
            z_i[j] = prior()
    return z_i

def perturbation_two(z, s=0.5):
    z_i = z.clone()
    for j in range(z.size()[0]):
        z_i[j] = np.random.normal(z.data[j])
    return z_i

def perturbation_three(z, s):
    z_i = z.clone()
    for j in range(z.size()[0]):
        if not flip(s):
            z_i[j] = np.random.normal(z.data[j])
    return z_i

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

    samples, Z = produce_samples(G, g_input_size, 1000)
    analysis(G, Z, samples)

