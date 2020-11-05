import argparse
import re
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from tqdm import tqdm
import expman

import seaborn as sns
sns.set_theme(style='whitegrid')

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

    formatter = lambda x, pos: f'{(x // 1000):g}k' if x >= 1000 else str(x)
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
            
            fig, axes = plt.subplots(2, 2)
            ax1, ax2, ax3, ax4 = axes.flat
            
            ax1.xaxis.set_major_formatter(formatter)
            ax2.xaxis.set_major_formatter(formatter)
            ax3.xaxis.set_major_formatter(formatter)
            ax4.xaxis.set_major_formatter(formatter)
            
            # generator losses
            gen = ['generator_encoder_loss', 'images_reconstruction_loss',
                   'latent_reconstruction_loss', 'generator_encoder_total_loss']
            train_log.plot(x='step', y=gen, logy='sym', ax=ax1)
            ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.2))
           
            # discriminator losses
            dis = ['discriminator_loss', 'gradient_penalty_loss', 'discriminator_total_loss']
            train_log.plot(x='step', y=dis, logy='sym', ax=ax3)
            ax3.legend(loc='upper left', bbox_to_anchor=(0.0, -0.2))
                
            # scores
            scores = ['real_score', 'fake_score']            
            train_log.plot(x='step', y=scores, logy='sym', ax=ax2)
            ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 1.2))
            
            # metrics
            metric_log.plot(x='step', y=['auc', 'balanced_accuracy'], ax=ax4)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.0, -0.2))
            
            # params
            params_str = exp.params.to_string()
            plt.figtext(0.0, 0.5, params_str, ha='right', va='center', family='monospace')
            
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
