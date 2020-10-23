import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_log(log):
    fig, axes = plt.subplots(3, 1)
    
    columns = (
        # generator losses
        ['generator_encoder_loss', 'images_reconstruction_loss', 'latent_reconstruction_loss',
         'generator_encoder_total_loss'],
   
        # discriminator losses
        ['discriminator_loss', 'gradient_penalty_loss', 'discriminator_total_loss'],
        
        # scores
        ['real_score', 'fake_score']
    )
    
    for ax, cols in zip(axes, columns):
        log.plot(y=cols, logy='sym', ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    return fig


def main(args):
    log = pd.read_csv(args.log_file)
    if log.empty:
        print('Empty log. Nothing to plot')
        return
        
    out = args.output if args.output else args.log_file.replace('.csv.gz', '.pdf')
    fig = plot_log(log)
    fig.savefig(out, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Stuff')
    parser.add_argument('log_file', type=str, help='Log file to plot')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file')
    
    args = parser.parse_args()
    main(args)
