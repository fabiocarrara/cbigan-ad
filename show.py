import argparse
import glob
import os
import re
import numpy as np
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

    exps = expman.gather(args.run).filter(args.filter)
    exps = exps.sort(key=lambda exp: exp.params.category)

    with PdfPages(args.output) as pdf:
        for exp in tqdm(exps):
            category = exp.params.category
            train_log = exp.path_to(f'log_{category}.csv.gz')
            train_log = pd.read_csv(train_log)
            
            metric_log = exp.path_to(f'metrics_{category}.csv')
            metric_log = pd.read_csv(metric_log)
            
            best_recon_file = exp.path_to(f'best_recon_{category}.png')
            best_recon = plt.imread(best_recon_file)

            last_recon_file = exp.path_to(f'last_recon_{category}.png')
            last_recon = plt.imread(last_recon_file)
            
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
            best_recon_step = train_log[(train_log.step % 1000) == 0].images_reconstruction_loss.idxmin()
            best_recon_step = train_log.loc[best_recon_step, 'step']
            ax4.axvline(best_recon_step, color='black', lw=1)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.0, -0.2))
            
            
            # best/last reconstruction
            hw = best_recon.shape[1] // 2
            recon = np.hstack((best_recon[:, :hw, :], last_recon[:, :hw, :]))
            ax5.imshow(recon)
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


def _get_scores(exps, best=False):
        
    metrics = exps.collect('metrics_*.csv')
    
    fixed_cols = metrics.nunique() == 1
    # do not consider the following as fixed params
    # (in the case there is only one run)
    fixed_cols.category = False
    fixed_cols.exp_id = False
    fixed_cols.exp_name = False
    
    fixed_params = metrics.loc[0, fixed_cols]
    
    # get variable cols & select the best based on metrics
    results = metrics.loc[:, ~fixed_cols]
    grouped = results.groupby('exp_id', sort=False)
    results_idx = grouped.auc.idxmax() if best else grouped.apply(lambda x: x.index[-1])
    results = results.loc[results_idx]

    # remove extra columns
    results = results.drop(columns=['exp_id', 'exp_name'])
    
    is_texture = results.category.isin(textures)
    results.loc[is_texture, 'type'] = 'texture'
    results.loc[~is_texture, 'type'] = 'object'

    results.type = pd.Categorical(results.type, ['texture', 'object', 'mean'])
    results.category = pd.Categorical(results.category, textures + objects + ['mean'])
    results = results.set_index(['type', 'category']).sort_index()

    table = results.pivot_table(values=['balanced_accuracy', 'auc'], columns=['type', 'category'])
    
    textures_mean = table['texture'].mean(axis=1) if 'texture' in table else None
    objects_mean = table['object'].mean(axis=1) if 'object' in table else None
    overall_mean = table.mean(axis=1)
    
    table.loc[:, ('texture', 'mean')] = textures_mean
    table.loc[:, ('object', 'mean')] = objects_mean
    table.loc[:, ('mean', 'mean')] = overall_mean
    
    table = table.sort_index(axis=1)
    
    return fixed_params, results, table


def print_scores(args):
    exps = expman.gather(args.run).filter(args.filter)
    
    fixed_params, results, table = _get_scores(exps, args.best)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 500):
      print('Common Parameters')
      print('=================')
      print(fixed_params)
      print('=================')
      print()
      print('Run Metrics')
      print('=================')
      print(results)
      print('=================')
      print()
      print('Best Metrics')
      print('=================')
      with pd.option_context('display.float_format', '{:.2f}'.format):
          print(table)
      print('=================')
      print()


def compare(args):
    
    exps1 = expman.gather(args.run1).filter(args.filter)
    exps2 = expman.gather(args.run2).filter(args.filter)
    
    fixed_params1, results1, table1 = _get_scores(exps1)
    fixed_params2, results2, table2 = _get_scores(exps2)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print('1]', args.run2)
            print(table2)
            print()
            print('2]', args.run1)
            print(table1)
            print()
        with pd.option_context('display.float_format', '{:.1%}'.format):
            print('D] {} - {}'.format(args.run2, args.run1))
            print(table2 - table1)
    

def compare_videos(args):
    exps = expman.gather(args.run).filter(args.filter)

    # get params and video paths
    params = exps.collect()
    params['video'] = [glob.glob(e.path_to('*.mp4'))[0] for e in exps]
    params = params.sort_values(['alpha', 'd_iter'], ascending=[False, True])

    video_paths = params.video.values
    video_labels = params.exp_name.values

    # find best grid aspect ratio
    n = len(video_paths)
    square_side = np.ceil(np.sqrt(n)).astype(int)
    nrows = np.arange(1, square_side + 1)
    ncols = np.ceil(n / nrows)
    aspect_ratios_per_nrows =  (6 / 7) * (ncols / nrows)
    best = np.argmin(np.abs(aspect_ratios_per_nrows - (16 / 9)))
    w = ncols[best].astype(int)
    h = nrows[best].astype(int)

    # build ffmpeg command
    input_args = [f'-i {path}' for path in video_paths]
    input_args = ' '.join(input_args)

    pad_w, pad_h = 10, 30

    filter_complex = [
        f'[{i}:v] '
        f'setpts=PTS-STARTPTS, '
        f'pad=width=iw+{pad_w}: height=ih+{pad_h}: x={pad_w // 2}: y={pad_h}, '
        f'drawtext=text=\'{label}\': fontcolor=white: fontsize=24: x=(w-tw)/2: y=({pad_h}-th)/2 '
        f'[a{i}]' for i, label in enumerate(video_labels)]

    xstack_inputs = ''.join(f'[a{i}]' for i in range(n))

    widths = ['0'] + [f'w{i}' for i in range(w - 1)]
    heights = ['0'] + [f'h{i}' for i in range(h - 1)]

    xstack_layout = [ '+'.join(widths[:i+1]) + '_' + '+'.join(heights[:j+1]) for j in range(h) for i in range(w) ]
    xstack_layout = '|'.join(xstack_layout)

    xstack_filter = f'{xstack_inputs}xstack=inputs={n}: layout={xstack_layout}[out]'

    filter_complex += [xstack_filter]
    filter_complex = ';'.join(filter_complex)

    cmd = f'ffmpeg {input_args} -filter_complex "{filter_complex}" -map "[out]" -c:v hevc -crf 23 -preset fast {args.output}'
    print(cmd)


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
    parser_score.add_argument('--best', action='store_true', default=False)
    parser_score.set_defaults(func=print_scores)
    
    parser_cmp = subparsers.add_parser('cmp')
    parser_cmp.add_argument('run1', default='runs/')
    parser_cmp.add_argument('run2', default='runs/')
    parser_cmp.set_defaults(func=compare)

    parser_vid = subparsers.add_parser('video')
    parser_vid.add_argument('run', default='runs/')
    parser_vid.add_argument('-o', '--output', default='vid_cmp.mp4')
    parser_vid.set_defaults(func=compare_videos)
    
    args = parser.parse_args()
    args.func(args)
