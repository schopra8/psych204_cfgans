# --------------------------------------------------------------------------------------------------
# Author: Sahil Chopra
# Psych 204, Fall 2017
# Goal: Given a Generator Model, load its weights and sample its worlds. 
#       When we find a sample that seems to produce a realistic world,
#       continue to perturb this vector for a 100 iterations according a perturbation format,
#       note the distribution of worlds that are produced.
# --------------------------------------------------------------------------------------------------

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to Generator Model Weights')
    args = parser.parse_args()

    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



def perturbation_one():
    # z_i' = flip(s) ? z_i : prior(z_i)  
    pass 

def perturbation_two():
    # z_i' = N(z_i, sigma)
    pass

def perturbation_three():
    # z_i' = flip(s) ? z_i : N(z_i, sigma)
    pass
