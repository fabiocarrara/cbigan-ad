import argparse

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

from sklearn import metrics
from tqdm import tqdm

import numpy as np
import pandas as pd

from mvtec_ad import textures, objects, get_train_data, get_test_data
from model import make_generator, make_encoder, make_discriminator
from losses import l1


def anomaly_score(generator, encoder, discriminator_features, images, patch_size, lambda_):
    
    def _anomaly_score(images):
        latent = encoder(images, training=False)  # E(x)
        reconstructed_images = generator(latent, training=False)  # G(E(x))

        features = discriminator_features([images, latent], training=False)  # f_D(x, E(x))
        reconstructed_features = discriminator_features([reconstructed_images, latent], training=False)  # f_D(G(E(x)), E(x))

        pixel_distance = l1(images, reconstructed_images)  # L_R
        features_distance = l1(features, reconstructed_features)  # L_f_D

        return (1 - lambda_) * pixel_distance + lambda_ * features_distance
    
    # compute in patches
    b, h, w, c = images.shape
    scores = [ _anomaly_score(images[:, y:y+patch_size, x:x+patch_size, :])
        for y in range(0, h, patch_size)
        for x in range(0, w, patch_size)
    ]

    scores = K.stack(scores, axis=1)  # b x num_patches
    scores = K.max(scores, axis=1)  # take the max score as anomaly score
    return scores


def get_discriminator_features_model(discriminator, layer=-65):
    feature_layer = discriminator.layers[layer]
    discriminator_features = Model(discriminator.input, feature_layer.output)
    return discriminator_features
    
    
def evaluate(generator, encoder, discriminator_features, test_dataset, test_labels, patch_size=64, lambda_=0.1):
    scores, labels = [], []

    good_label = test_labels.index('good')
    def binarize_labels(labels):
        return labels == good_label

    for batch_images, batch_labels in tqdm(test_dataset):
        scores.append( anomaly_score(generator, encoder, discriminator_features, batch_images, patch_size, lambda_).numpy() )
        labels.append( binarize_labels(batch_labels).numpy() )

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    
    fpr, tpr, thr = metrics.roc_curve(labels, scores)
    balanced_accuracy = np.max((tpr + (1 - fpr)) / 2)
    auc = metrics.auc(fpr, tpr)
    
    return auc, balanced_accuracy


def main(args):
    # get data
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
    
    # checkpointer
    checkpoint = tf.train.Checkpoint(generator=generator, encoder=encoder, discriminator=discriminator)
    checkpoint.read(f'ckpt/{args.category}/ckpt_{args.category}_best').expect_partial()
                                     
    discriminator_features = get_discriminator_features_model(discriminator)
    auc, balanced_accuracy = evaluate(generator, encoder, discriminator_features,
                                      test_dataset, test_labels,
                                      patch_size=args.patch_size, lambda_=args.lambda_)
    print(f'{args.category}: AUC={auc}, BalAcc={balanced_accuracy}')


if __name__ == '__main__':

    categories = textures + objects
    parser = argparse.ArgumentParser(description='Score MVTec AD Test Datasets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data params
    parser.add_argument('category', type=str, choices=categories, help='MVTec-AD item category')
    parser.add_argument('--image-size', type=int, default=128, help='Resize image to this size')
    parser.add_argument('--patch-size', type=int, default=128, help='Extract patches of this size')
    
    # model params
    parser.add_argument('--latent-size', type=int, default=64, help='Latent variable dimensionality')
    parser.add_argument('--channels', type=int, default=3, help='Multiplier for the number of channels in Conv2D layers')
    
    parser.add_argument('--ge-up', type=str, choices=('bilinear', 'transpose'), default='bilinear', help='Upsampling method to use in G')
    parser.add_argument('--ge-bn', type=str, choices=('batch', 'layer', 'instance', 'none'), default='none', help="Whether to use Normalization in G and E")
    parser.add_argument('--ge-act', type=str, choices=('relu', 'lrelu'), default='lrelu', help='Activation to use in G and E')
    parser.add_argument('--d-bn', type=str, choices=('batch', 'layer', 'instance', 'none'), default='none', help="Whether to use Normalization in D")
    parser.add_argument('--d-act', type=str, choices=('relu', 'lrelu'), default='lrelu', help='Activation to use in D')

    
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')    
    parser.add_argument('--lambda', type=float, dest='lambda_', default=0.1, help='weight of discriminator features when scoring')
    
    args = parser.parse_args()
    main(args)
