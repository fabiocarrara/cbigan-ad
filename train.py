import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # do not print tf INFO messages

import argparse

import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras import layers as L
from tensorflow.keras import optimizers as O

from expman import Experiment
from tqdm import tqdm

import imageio
import numpy as np
import pandas as pd

from mvtec_ad import textures, objects, get_train_data, get_test_data
from model import make_generator, make_encoder, make_discriminator
from losses import train_step
from score import get_discriminator_features_model, evaluate
from util import VideoSaver

def main(args):

    # do not track lambda param, it can be changed after train
    exp = Experiment(args, ignore=('lambda_',))
    print(exp)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # get data
    train_dataset = get_train_data(args.category,
                                   image_size=args.image_size,
                                   patch_size=args.patch_size,
                                   batch_size=args.batch_size,
                                   n_batches=args.n_batches,
                                   rotation_range=args.rotation_range,
                                   seed=args.seed)

    test_dataset, test_labels = get_test_data(args.category,
                                              image_size=args.image_size,
                                              patch_size=args.patch_size,
                                              batch_size=args.batch_size)

    is_object = args.category in objects

    # build models
    generator = make_generator(args.latent_size, channels=args.channels,
                               upsample_first=is_object, upsample_type=args.ge_up,
                               bn=args.ge_bn, act=args.ge_act)
    encoder = make_encoder(args.patch_size, args.latent_size, channels=args.channels,
                           bn=args.ge_bn, act=args.ge_act)
    discriminator = make_discriminator(args.patch_size, args.latent_size, channels=args.channels,
                                       bn=args.d_bn, act=args.d_act)
    # feature extractor model for evaluation
    discriminator_features = get_discriminator_features_model(discriminator)
    
    # build optimizers
    generator_encoder_optimizer = O.Adam(args.lr, beta_1=args.ge_beta1, beta_2=args.ge_beta2)
    discriminator_optimizer = O.Adam(args.lr, beta_1=args.d_beta1, beta_2=args.d_beta2)

    # for smoothing generator and encoder evolution
    ema = tf.train.ExponentialMovingAverage(decay=args.ge_decay)

    # checkpointer
    checkpoint = tf.train.Checkpoint(generator=generator, encoder=encoder,
                                     discriminator=discriminator,
                                     generator_encoder_optimizer=generator_encoder_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)
    best_ckpt_path = exp.ckpt(f'ckpt_{args.category}_best')
    last_ckpt_path = exp.ckpt(f'ckpt_{args.category}_last')

    # log stuff
    log, log_file = exp.require_csv(f'log_{args.category}.csv.gz')
    metrics, metrics_file = exp.require_csv(f'metrics_{args.category}.csv')
    best_metric = 0.
    best_recon = float('inf')
    best_recon_file = exp.path_to(f'best_recon_{args.category}.png')
    last_recon_file = exp.path_to(f'last_recon_{args.category}.png')

    # animate generation during training
    n_preview = 6
    train_batch = next(iter(train_dataset))[:n_preview]
    test_batch = next(iter(test_dataset))[0][:n_preview]
    latent_batch = tf.random.normal([n_preview, args.latent_size])

    if not is_object:  # take random patches from test images
        patch_location = np.random.randint(0, args.image_size - args.patch_size, (n_preview, 2))
        test_batch = [x[i:i+args.patch_size, j:j+args.patch_size, :]
                      for x, (i, j) in zip(test_batch, patch_location)]
        test_batch = K.stack(test_batch)

    video_out = exp.path_to(f'{args.category}.mp4')
    video_options = dict(fps=30, codec='libx265', quality=4)  # see imageio FFMPEG options
    video_saver = VideoSaver(train_batch, test_batch, latent_batch, video_out, **video_options)
    video_saver.generate_and_save(generator, encoder)

    # train loop
    progress = tqdm(train_dataset, desc=args.category, dynamic_ncols=True)
    try:
        for step, image_batch in enumerate(progress, start=1):
            if step == 1 or args.d_iter == 0:  # only for JIT compilation (tf.function) to work
                d_train = True
                ge_train = True
            else:
                n_iter = step % (abs(args.d_iter) + 1)  # can be in [0, d_iter]
                d_train = (n_iter != 0) if (args.d_iter > 0) else (n_iter == 0)  # True in [1, d_iter]
                ge_train = not d_train  # True when step == d_iter + 1

            losses, scores = train_step(image_batch, generator, encoder, discriminator,
                                        generator_encoder_optimizer, discriminator_optimizer,
                                        d_train, ge_train, alpha=args.alpha, gp_weight=args.gp_weight)

            if (args.ge_decay > 0) and (step % 10 == 0):
                ge_vars = generator.trainable_variables + encoder.trainable_variables
                ema.apply(ge_vars)  # update exponential moving average
                for v in ge_vars:  # assign to each variable its exponential moving average
                    v = ema.average(v)

            # tensor to numpy
            losses = {n: l.numpy() if l is not None else l for n, l in losses.items()}
            scores = {n: s.numpy() if s is not None else s for n, s in scores.items()}

            # log step metrics
            entry = {'step': step, 'timestamp': pd.to_datetime('now'), **losses, **scores}
            log = log.append(entry, ignore_index=True)

            if step % 100 == 0:
                preview = video_saver.generate_and_save(generator, encoder)
                
            if step % 1000 == 0:
                log.to_csv(log_file, index=False)
                checkpoint.write(file_prefix=last_ckpt_path)
                
                auc, balanced_accuracy = evaluate(generator, encoder, discriminator_features,
                                                  test_dataset, test_labels,
                                                  patch_size=args.patch_size, lambda_=args.lambda_)
                                                  
                entry = {'step': step, 'auc': auc, 'balanced_accuracy': balanced_accuracy}         
                metrics = metrics.append(entry, ignore_index=True)
                metrics.to_csv(metrics_file, index=False)
                
                if auc > best_metric:
                    best_metric = auc
                    checkpoint.write(file_prefix=best_ckpt_path)
                
                # save last image to inspect it during training
                imageio.imwrite(last_recon_file, preview)
                
                recon = losses['images_reconstruction_loss']
                if recon < best_recon:
                    best_recon = recon
                    imageio.imwrite(best_recon_file, preview)
                
                progress.set_postfix({
                    'AUC': f'{auc:.1%}',
                    'BalAcc': f'{balanced_accuracy:.1%}',
                    'BestAUC': f'{best_metric:.1%}',
                })

    except KeyboardInterrupt:
        checkpoint.write(file_prefix=last_ckpt_path)
    finally:
        log.to_csv(log_file, index=False)
        video_saver.close()
        
    # score the test set
    checkpoint.read(best_ckpt_path)
    
    auc, balanced_accuracy = evaluate(generator, encoder, discriminator_features,
                                      test_dataset, test_labels,
                                      patch_size=args.patch_size, lambda_=args.lambda_)
    print(f'{args.category}: AUC={auc}, BalAcc={balanced_accuracy}')

if __name__ == '__main__':

    categories = textures + objects
    parser = argparse.ArgumentParser(description='Train CBiGAN on MVTec AD',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data params
    parser.add_argument('category', type=str, choices=categories, help='MVTec-AD item category')
    parser.add_argument('--image-size', type=int, default=128, help='Resize image to this size')
    parser.add_argument('--patch-size', type=int, default=128, help='Extract patches of this size')
    parser.add_argument('--rotation-range', type=int, nargs=2, default=(0, 0), help='Random rotation range in degrees')
    
    # model params
    parser.add_argument('--latent-size', type=int, default=64, help='Latent variable dimensionality')
    parser.add_argument('--channels', type=int, default=3, help='Multiplier for the number of channels in Conv2D layers')
    
    parser.add_argument('--ge-decay', type=float, default=0.999, help='Moving average decay for paramteres of G and E')
    parser.add_argument('--ge-up', type=str, choices=('bilinear', 'transpose'), default='bilinear', help='Upsampling method to use in G')
    parser.add_argument('--ge-bn', type=str, choices=('batch', 'layer', 'instance', 'none'), default='none', help="Whether to use Normalization in G and E")
    parser.add_argument('--ge-act', type=str, choices=('relu', 'lrelu'), default='lrelu', help='Activation to use in G and E')
    parser.add_argument('--d-bn', type=str, choices=('batch', 'layer', 'instance', 'none'), default='none', help="Whether to use Normalization in D")
    parser.add_argument('--d-act', type=str, choices=('relu', 'lrelu'), default='lrelu', help='Activation to use in D')
    
    # optimizer params
    parser.add_argument('--n-batches', type=int, default=200_000, help='Number of batches processed in the training phase')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--ge-beta1', type=float, default=0, help='Beta_1 value of Adam for G and E')
    parser.add_argument('--ge-beta2', type=float, default=0.099, help='Beta_2 value of Adam for G and E')
    parser.add_argument('--d-beta1', type=float, default=0, help='Beta_1 value of Adam for D')
    parser.add_argument('--d-beta2', type=float, default=0.909, help='Beta_2 value of Adam for D')
    
    parser.add_argument('--alpha', type=float, default=1e-4, help='Consistency loss weight')
    parser.add_argument('--gp-weight', type=float, default=2.5, help='Gradient penalty weight')
    
    parser.add_argument('--d-iter', type=int, default=1, help='Number of times D trains more than G and E (or viceversa using negative values; 0 = simultaneous step)')
    
    # other parameters
    parser.add_argument('--lambda', type=float, dest='lambda_', default=0.1, help='weight of discriminator features when scoring')
    parser.add_argument('--seed', type=int, default=42, help='rng seed')
    
    args = parser.parse_args()
    main(args)
