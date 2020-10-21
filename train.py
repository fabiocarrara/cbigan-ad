import argparse

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers as L
from tensorflow.keras import optimizers as O

# from sklearn import metrics
from tqdm import tqdm

import numpy as np
import pandas as pd

from mvtec_ad import textures, objects, get_data
from model import make_generator, make_encoder, make_discriminator
from losses import train_step
from util import VideoSaver
    
def train(args):

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # get data
    train_dataset, test_dataset = get_data(args.category,
                                           image_size=args.image_size,
                                           patch_size=args.patch_size,
                                           batch_size=args.batch_size, 
                                           n_batches=args.n_batches,
                                           rotation_range=args.rotation_range,
                                           seed=args.seed)
    
    is_object = args.category in objects
    
    # build models
    generator = make_generator(args.latent_size, channels=args.channels,
                               upsample_first=is_object, upsample_type=args.ge_up,
                               bn=args.ge_bn, act=args.ge_act)
    encoder = make_encoder(args.patch_size, args.latent_size, channels=args.channels,
                           bn=args.ge_bn, act=args.ge_act)
    discriminator = make_discriminator(args.patch_size, args.latent_size, channels=args.channels,
                                       bn=args.d_bn, act=args.d_act)
    # build optimizers
    generator_encoder_optimizer = O.Adam(args.lr, beta_1=args.ge_beta1, beta_2=args.ge_beta2)
    discriminator_optimizer = O.Adam(args.lr, beta_1=args.d_beta1, beta_2=args.d_beta2)
    
    # checkpointer
    checkpoint = tf.train.Checkpoint(generator=generator, encoder=encoder,
                                     discriminator=discriminator,
                                     generator_encoder_optimizer=generator_encoder_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)
                                     
    # log stuff
    log = pd.DataFrame()
    n_preview = 6
    train_batch = next(iter(train_dataset))[:n_preview]
    test_batch = next(iter(test_dataset))[0][:n_preview]
    latent_batch = tf.random.normal([n_preview, args.latent_size])
    
    if not is_object:
        patch_location = np.random.randint(0, image_size - patch_size, (n_preview, 2))
        train_batch = [x[i:i+patch_size, j:j+patch_size, :]
                      for x, (i, j) in zip(train_batch, patch_location)]
        test_batch = [x[i:i+patch_size, j:j+patch_size, :]
                      for x, (i, j) in zip(test_batch, patch_location)]
                      
        train_batch = K.stack(train_batch)
        test_batch = K.stack(test_batch)
        
    best_metric = float('inf')
    video_out = f'{args.category}.mp4'
    video_saver = VideoSaver(train_batch, test_batch, latent_batch, video_out)
    
    # train loop
    try:
        image_reconstruction_loss = []    
        for step, image_batch in enumerate(tqdm(train_dataset)):
            if step == 0:  # only for tf.function to work
                d_train = True
                ge_train = True
            else:
                d_train = (step % (args.d_iter + 1)) != args.d_iter  # True in [0, d_iter-1]
                ge_train = not d_train  # True when step == d_iter
                
            losses, scores = train_step(image_batch, generator, encoder, discriminator,
                                        generator_encoder_optimizer, discriminator_optimizer,
                                        d_train, ge_train, alpha=args.alpha, gp_weight=args.gp_weight)
            # tensor to numpy
            losses = {n: l.numpy() if l is not None else l for n, l in losses.items()}
            scores = {n: s.numpy() if s is not None else s for n, s in scores.items()}
            
            # log step metrics
            entry = {'step': step, 'timestamp': pd.to_datetime('now'), **losses, **scores}
            log = log.append(entry, ignore_index=True)

            image_reconstruction_loss.append(losses['images_reconstruction_loss'])

            if (step + 1) % 100 == 0:
                video_saver.generate_and_save(generator, encoder)
            
            if (step + 1) % 1000 == 0:
                log.to_csv(f'log_{args.category}.csv', index=False)
                checkpoint.save(file_prefix=f'ckpt/{args.category}/ckpt_{args.category}_iter{step}')
                
                mean_image_difference = np.mean(image_reconstruction_loss)
                if mean_image_difference < best_metric:
                    best_metric = mean_image_difference
                    checkpoint.write(file_prefix=f'ckpt/{args.category}/ckpt_{args.category}_best')
                    image_reconstruction_loss.clear()
    except KeyboardInterrupt:
        checkpoint.save(file_prefix=f'ckpt/{args.category}/ckpt_{args.category}_last')
    finally:
        log.to_csv(f'log_{args.category}.csv', index=False)
        video_saver.close()


if __name__ == '__main__':

    # utility for argparse
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


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
    
    parser.add_argument('--ge-up', type=str, choices=('bilinear', 'transpose'), default='bilinear', help='Upsampling method to use in G')
    parser.add_argument('--ge-bn', type=str2bool, nargs='?', const=True, default=False, help="Whether to use BatchNorm in G and E")
    parser.add_argument('--ge-act', type=str, choices=('relu', 'lrelu'), default='lrelu', help='Activation to use in G and E')
    parser.add_argument('--d-bn', type=str2bool, nargs='?', const=True, default=False, help="Whether to use BatchNorm in D")
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
    
    parser.add_argument('--d-iter', type=int, default=1, help='Number of times D trains more than G and E')
    
    # other parameters
    parser.add_argument('--lambda', type=float, default=0.1, help='weight of discriminator features when scoring')
    parser.add_argument('--seed', type=int, default=42, help='rng seed')
    
    args = parser.parse_args()
    train(args)