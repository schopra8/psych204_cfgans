import matplotlib.pyplot as plt
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dfile', help='discriminator losses file', default='./losses/d_losses_100000.txt')
    parser.add_argument('--gfile', help='generator losses file', default='./losses/g_losses_100000.txt')
    args = parser.parse_args()

    with open(args.dfile,"rb") as fp1:
        d_losses = pickle.load(fp1)

        with open(args.gfile,"rb") as fp2:
            g_losses = pickle.load(fp2)

            # Just a figure and one subplot
            fig, ax = plt.subplots()
            num_epochs = len(d_losses)
            ax.plot(range(1, num_epochs + 1), d_losses, 'r', label="Discriminator")
            ax.plot(range(first_discrim_epochs + 1, num_epochs + 1), g_losses, 'b', label="Generator")
            plt.title('Loss vs. Time')
            plt.legend()
            fig.savefig('loss_plot{}'.format(args.data_size))
