import argparse
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


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


class VideoSaver (object):

    def __init__(self, train_samples, test_samples, latent_samples, out_file, error_cm='viridis', **writer_kw):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.latent_samples = latent_samples
        
        self.writer = imageio.get_writer(out_file, **writer_kw)
        self.error_colormap = tf.constant(plt.get_cmap(error_cm).colors)
    
    def generate_and_save(self, gen, enc):
        reconstructed_train_samples = gen(enc(self.train_samples, training=False), training=False)
        reconstructed_test_samples = gen(enc(self.test_samples, training=False), training=False)
        generated_samples = gen(self.latent_samples, training=False)

        # build error maps in [0, 255]
        errormap_train_samples = 255 * K.mean(0.5 * K.abs(reconstructed_train_samples - self.train_samples), axis=-1)
        errormap_test_samples = 255 * K.mean(0.5 * K.abs(reconstructed_test_samples - self.test_samples), axis=-1)
        # convert to uint8 (using int32 instead for pleasing take())
        errormap_train_samples = K.cast(errormap_train_samples, dtype='int32')
        errormap_test_samples = K.cast(errormap_test_samples, dtype='int32')
        # apply colormap: result is float in [0,1]
        errormap_train_samples = tf.experimental.numpy.take(self.error_colormap, errormap_train_samples, axis=0)
        errormap_test_samples = tf.experimental.numpy.take(self.error_colormap, errormap_test_samples, axis=0)
        # rescale to [-1, 1]
        errormap_train_samples = 2 * errormap_train_samples - 1
        errormap_test_samples = 2 * errormap_test_samples - 1

        rows = [
            self.train_samples,
            reconstructed_train_samples,
            errormap_train_samples,
            self.test_samples,
            reconstructed_test_samples,
            errormap_test_samples,
            generated_samples
        ]
        
        # concatenate images horizontally in rows
        rows = [K.concatenate(tf.unstack(row, axis=0), axis=1) for row in rows]
        # concatenate rows vertically in a single image
        image = K.concatenate(rows, axis=0).numpy()
        
        # convert [-1, 1] to [0, 255]
        image = ((image + 1) * 127.5).astype(np.uint8)
        self.writer.append_data(image)
        
        return image
    
    def close(self):
        self.writer.close()
        
        


