import argparse
import expman

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
    
    def _reconstruction_errors(images):
        latent = encoder(images, training=False)  # E(x)
        reconstructed_images = generator(latent, training=False)  # G(E(x))

        features = discriminator_features([images, latent], training=False)  # f_D(x, E(x))
        reconstructed_features = discriminator_features([reconstructed_images, latent], training=False)  # f_D(G(E(x)), E(x))

        pixel_distance = l1(images, reconstructed_images)  # L_R
        features_distance = l1(features, reconstructed_features)  # L_f_D

        return pixel_distance, features_distance

    def _anomaly_score(pixel_distance, features_distance, lambda_):
        return (1 - lambda_) * pixel_distance + lambda_ * features_distance
    
    # compute in patches
    b, h, w, c = images.shape
    distances = [ _reconstruction_errors(images[:, y:y+patch_size, x:x+patch_size, :])
        for y in range(0, h, patch_size)
        for x in range(0, w, patch_size)
    ]

    pixel_distances, features_distances = K.stack(distances, axis=2)  # 2 x b x num_patches
    scores = [_anomaly_score(pixel_distances, features_distances, lamb) for lamb in lambda_] # l x b x num_patches
    scores = K.max(scores, axis=2)  # take the max score among patches as anomaly score
    return scores


def get_discriminator_features_model(discriminator, layer=-65):
    feature_layer = discriminator.layers[layer]
    discriminator_features = Model(discriminator.input, feature_layer.output)
    return discriminator_features
    
    
def evaluate(generator, encoder, discriminator_features, test_dataset, test_labels, patch_size=64, lambda_=(0.1,)):
    return_scalar = False
    if not isinstance(lambda_, (tuple, list)):
        lambda_ = (lambda_,)
        return_scalar = True

    scores, labels = [], []

    good_label = test_labels.index('good')
    def binarize_labels(labels):
        return labels != good_label

    for batch_images, batch_labels in tqdm(test_dataset, leave=False):
        scores.append( anomaly_score(generator, encoder, discriminator_features, batch_images, patch_size, lambda_).numpy() )
        labels.append( binarize_labels(batch_labels).numpy() )

    scores = np.concatenate(scores, axis=1) # l x n
    labels = np.concatenate(labels) # n

    auc, balanced_accuracy = [], []
    for s in scores:
        fpr, tpr, thr = metrics.roc_curve(labels, s)
        balanced_accuracy.append( np.max((tpr + (1 - fpr)) / 2) )
        auc.append( metrics.auc(fpr, tpr) )

    auc = auc[0] if return_scalar else auc
    balanced_accuracy = balanced_accuracy[0] if return_scalar else balanced_accuracy

    return auc, balanced_accuracy


def main(args):

    exp = expman.from_dir(args.run)
    params = exp.params

    batch_size = args.batch_size if args.batch_size else params.batch_size
    is_object = params.category in objects

    # get data
    test_dataset, test_labels = get_test_data(params.category,
                                              image_size=params.image_size,
                                              patch_size=params.patch_size,
                                              batch_size=batch_size)
    
    # build models
    generator = make_generator(params.latent_size,
                               channels=params.channels,
                               upsample_first=is_object,
                               upsample_type=params.ge_up,
                               bn=params.ge_bn,
                               act=params.ge_act)

    encoder = make_encoder(params.patch_size,
                           params.latent_size,
                           channels=params.channels,
                           bn=params.ge_bn,
                           act=params.ge_act)

    discriminator = make_discriminator(params.patch_size,
                                       params.latent_size,
                                       channels=params.channels,
                                       bn=params.d_bn,
                                       act=params.d_act)
    
    # checkpointer
    checkpoint = tf.train.Checkpoint(generator=generator, encoder=encoder, discriminator=discriminator)
    ckpt_suffix = 'best' if args.best else 'last'
    ckpt_path = exp.path_to(f'ckpt/ckpt_{params.category}_{ckpt_suffix}')
    checkpoint.read(ckpt_path).expect_partial()
                                     
    discriminator_features = get_discriminator_features_model(discriminator)
    auc, balanced_accuracy = evaluate(generator, encoder, discriminator_features,
                                      test_dataset, test_labels,
                                      patch_size=params.patch_size, lambda_=args.lambda_)

    # print(f'{params.category}: AUC={auc}, BalAcc={balanced_accuracy}')
    index = pd.Index(args.lambda_, name='lambda')
    table = pd.DataFrame({'auc': auc, 'balanced_accuracy': balanced_accuracy}, index=index)
    print(table)


if __name__ == '__main__':
    default_lambdas = (0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.)
    categories = textures + objects
    parser = argparse.ArgumentParser(description='Score MVTec AD Test Datasets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('run', help='path to run dir')

    # model params
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--best', action='store_true', default=False, help='whether to use the early stopped model')
    parser.add_argument('--lambda', type=float, dest='lambda_', nargs='+', default=default_lambdas, help='weight of discriminator features when scoring')
    
    args = parser.parse_args()
    main(args)
