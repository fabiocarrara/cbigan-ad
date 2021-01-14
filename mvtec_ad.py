import os
import math

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.layers.experimental import preprocessing as P
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data.experimental import AUTOTUNE

from tqdm import tqdm


textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
objects = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

def random_rotation_crop_no_edges(image, rotation_range, crop_size):
    # Randomly rotates image, then crops out the edges, then performs random crop.
    # Adapted from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

    def _find_central_fraction_with_no_edges(w, h, angle):
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
        delta = math.pi - alpha - gamma

        length = h if (w < h) else w
        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        cw = bb_w - 2 * x
        ch = bb_h - 2 * y

        return (cw * ch) / (h * w)

    b, h, w, c = image.shape

    angle = 2. * np.pi * np.random.uniform(rotation_range[0], rotation_range[1])
    rotated = tfa.image.rotate(image, angle, interpolation='BILINEAR')
    central_fraction = _find_central_fraction_with_no_edges(w, h, angle)
    noedges = tf.image.central_crop(rotated, central_fraction)
    cropped = tf.image.random_crop(noedges, (b, crop_size, crop_size, c))
    return cropped


def get_train_data(category, batch_size=32, image_size=128, patch_size=128, rotation_range=(0,0), n_batches=50_000, seed=42):
    train_data_dir = f'data/mvtec-ad/{category}/train/'

    dataset_kwargs = dict(image_size=(image_size, image_size), batch_size=1, shuffle=False, seed=seed)
    rescale = P.Rescaling(scale=1./127.5, offset=-1)  # scale from [0, 255] to [-1, 1]

    ### train dataset loading
    os.makedirs('cache', exist_ok=True)
    cache_file = f'cache/{category}_train_dataset_i{image_size}_p{patch_size}_r{rotation_range[0]}-{rotation_range[1]}_s{seed}.npy'

    if not os.path.exists(cache_file):  # check cache (only texture are cached)
        # load and resize images
        train_dataset = image_dataset_from_directory(train_data_dir, label_mode=None, **dataset_kwargs)
        n_train_images = len(train_dataset)
        
        # scale pixel range
        train_dataset = train_dataset.map(rescale, num_parallel_calls=AUTOTUNE)
        
        # cache the rescaled dataset
        train_dataset = train_dataset.unbatch().cache()

        """
        # compute mean image
        mean_image = K.zeros((image_size, image_size, 3))
        mean_image = train_dataset.reduce(mean_image, lambda x, y: x + y)
        mean_image /= n_train_images
        """

        # replicate dataset to get roughly 50000 augmented images
        num_repeats = 50000 // n_train_images
        n_total = num_repeats * n_train_images
        train_dataset = train_dataset.repeat(num_repeats)

        # apply random rotation
        if category in textures:  # for textures: rotate and crop without edges
            def augmentation(x):  # slow computation, we will cache this
                return tf.py_function(random_rotation_crop_no_edges,
                                      inp=[x, rotation_range, patch_size], Tout=tf.float32)
        else:  # for objects: rotate (reflect edges), no crop
            factor = (rotation_range[0] / 360., rotation_range[1] / 360.)
            # fill_mode='nearest' is available only in tf-nightly (2.4.0)
            augmentation = P.RandomRotation(factor, fill_mode='nearest', seed=seed)

        train_dataset = train_dataset.shuffle(10000).batch(32)
        train_dataset = train_dataset.map(augmentation, num_parallel_calls=AUTOTUNE)
        train_dataset = train_dataset.unbatch()

        if category in textures: # for textures: cache the augmented dataset
            print('Creating cache for dataset:', cache_file)
            train_dataset = tqdm(train_dataset.as_numpy_iterator(), total=n_total)
            train_dataset = np.stack(list(train_dataset))
            np.save(cache_file, train_dataset)
            train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    else:
        print('Loading dataset from cache:', cache_file)
        train_dataset = np.load(cache_file)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)

    ### shuffle, batch and prefetch
    train_dataset = train_dataset.repeat().shuffle(10000).batch(batch_size).take(n_batches)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset


def get_test_data(category, batch_size=32, image_size=128, patch_size=128):
    test_data_dir = f'data/mvtec-ad/{category}/test/'

    dataset_kwargs = dict(image_size=(image_size, image_size), batch_size=batch_size, shuffle=False)
    rescale = P.Rescaling(scale=1./127.5, offset=-1)  # scale from [0, 255] to [-1, 1]
    
    ### test dataset
    # load and resize images (load also labels for test dataset)
    test_dataset = image_dataset_from_directory(test_data_dir, **dataset_kwargs)
    test_labels = test_dataset.class_names

    # rescale pixel range, cache
    test_dataset = test_dataset.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return test_dataset, test_labels

    
if __name__ == '__main__':
    train_ds = get_train_data('leather', image_size=512, patch_size=64, rotation_range=(0,45))
    sample_batch = next(iter(train_ds))
    print(sample_batch.shape)
    
    test_ds, test_labels = get_test_data('leather', image_size=512, patch_size=64)
    sample_batch, label_batch = next(iter(test_ds))
    print(sample_batch.shape, label_batch.shape)

