import argparse
import re
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, AutoMinorLocator

from tqdm import tqdm
import expman

import seaborn as sns
sns.set_theme(style='darkgrid')

# avoid importing mvtec_ad and load tf for mere plotting
# from mvtec_ad import textures, objects
textures = [
    'carpet',
    'grid',
    'leather',
    'tile',
    'wood'
]

objects = [
    'bottle',
    'cable',
    'capsule',
    'hazelnut',
    'metal_nut',
    'pill',
    'screw',
    'toothbrush',
    'transistor',
    'zipper'
]


def plot_log(args):

    formatter = lambda x, pos: f'{(x // 1000):g}k' if x >= 1000 else f'{x:g}'
    formatter = FuncFormatter(formatter)

    exps = expman.gather(args.run)
    exps = expman.filter(args.filter, exps)
    exps = sorted(exps, key=lambda exp: exp.params.category)
    
    with PdfPages(args.output) as pdf:
        for exp in tqdm(exps):
            category = exp.params.category
            train_log = exp.path_to(f'log_{category}.csv.gz')
            train_log = pd.read_csv(train_log)
            
            metric_log = exp.path_to(f'metrics_{category}.csv')
            metric_log = pd.read_csv(metric_log)
            
            best_recon_file = exp.path_to(f'best_recon_{category}.png')
            best_recon = plt.imread(best_recon_file)
            
            # prepare figure
            zoom = 0.7
            fig = plt.figure(figsize=(20 * zoom, 8 * zoom))
            # gridspec and axes for plots
            gs = fig.add_gridspec(ncols=2, nrows=2, hspace=0.05, wspace=0.05, right=0.5)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax3 = fig.add_subplot(gs[1,0], sharex=ax1)
            ax4 = fig.add_subplot(gs[1,1], sharex=ax2)
            # gridspec and axes for preview images
            gs2 = fig.add_gridspec(ncols=1, nrows=2, hspace=0, wspace=0, left=0.55)
            ax5 = fig.add_subplot(gs2[:,0])
            # ticklabels format
            ax1.xaxis.set_major_formatter(formatter)
            ax2.xaxis.set_major_formatter(formatter)
            ax3.xaxis.set_major_formatter(formatter)
            ax4.xaxis.set_major_formatter(formatter)
            ax5.axis('off')
            # minor ticks position
            ax3.xaxis.set_minor_locator(AutoMinorLocator())
            ax4.xaxis.set_minor_locator(AutoMinorLocator())
            # minor grid style
            ax1.grid(b=True, which='minor', linewidth=0.5)
            ax2.grid(b=True, which='minor', linewidth=0.5)
            ax3.grid(b=True, which='minor', linewidth=0.5)
            ax4.grid(b=True, which='minor', linewidth=0.5)
            # right y-axes
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax4.yaxis.set_label_position("right")
            ax4.yaxis.tick_right()
            
            # generator losses
            gen = ['generator_encoder_loss', 'images_reconstruction_loss',
                   'latent_reconstruction_loss', 'generator_encoder_total_loss']
            train_log.plot(x='step', y=gen, logy='sym', ax=ax1)
            ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0))
           
            # discriminator losses
            dis = ['discriminator_loss', 'gradient_penalty_loss', 'discriminator_total_loss']
            train_log.plot(x='step', y=dis, logy='sym', ax=ax3)
            ax3.legend(loc='upper left', bbox_to_anchor=(0.0, -0.2))
                
            # scores
            scores = ['real_score', 'fake_score']            
            train_log.plot(x='step', y=scores, logy='sym', ax=ax2)
            ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0))
            
            # metrics
            metric_log.plot(x='step', y=['auc', 'balanced_accuracy'], ax=ax4)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.0, -0.2))
            
            # best reconstruction
            ax5.imshow(best_recon)
            ax5.margins(0)
            
            # params
            params_str = exp.params.to_string()
            plt.figtext(0.08, 0.5, params_str, ha='right', va='center', family='monospace')
            
            # save sigle figure in exp folder
            log_pdf = exp.path_to(f'log_{category}.pdf')
            fig.savefig(log_pdf, bbox_inches='tight')

            # add as page in PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def print_scores(args):
    
    exps = expman.gather(args.run)
    exps = expman.filter(args.filter, exps)
    
    metrics = expman.collect_all(exps, 'metrics_*.csv')
    
    print('Common Parameters')
    print('=================')
    fixed_cols = metrics.nunique() == 1
    fixed_params = metrics.loc[0, fixed_cols]
    print(fixed_params)
    print('=================')
    print()
    
    print('Run Metrics')
    print('=================')
    # get variable cols & select the best based on metrics
    results = metrics.loc[:, ~fixed_cols]
    best = results.groupby('exp_id').auc.idxmax()
    results = results.loc[best]
    
    # remove extra columns
    results = results.drop(columns=['exp_id', 'exp_name'])
    
    is_texture = results.category.isin(textures)
    results.loc[is_texture, 'type'] = 'texture'
    results.loc[~is_texture, 'type'] = 'object'

    results.type = pd.Categorical(results.type, ['texture', 'object', 'mean'])
    results.category = pd.Categorical(results.category, textures + objects + ['mean'])
    results = results.set_index(['type', 'category']).sort_index()

    print(results)
    print('=================')
    print()
    
    print('Best Metrics')
    print('=================')
    table = results.pivot_table(values=['balanced_accuracy', 'auc'], columns=['type', 'category'])
    
    textures_mean = table['texture'].mean(axis=1)
    objects_mean = table['object'].mean(axis=1)
    overall_mean = table.mean(axis=1)
    
    table.loc[:, ('texture', 'mean')] = textures_mean
    table.loc[:, ('object', 'mean')] = objects_mean
    table.loc[:, ('mean', 'mean')] = overall_mean
    
    table = table.sort_index(axis=1)
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(table)
    print('=================')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show Stuff')
    parser.add_argument('-f', '--filter', default={}, type=expman.exp_filter)
    subparsers = parser.add_subparsers()

    parser_log = subparsers.add_parser('log', description='Generate PDF of logs and metrics')
    parser_log.add_argument('run', default='runs/')
    parser_log.add_argument('-o', '--output', default='logs.pdf')
    parser_log.set_defaults(func=plot_log)
    
    parser_score = subparsers.add_parser('score')
    parser_score.add_argument('run', default='runs/')
    parser_score.set_defaults(func=print_scores)
    
    args = parser.parse_args()
    args.func(args)
