import imageio
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class VideoSaver (object):

    def __init__(self, train_samples, test_samples, latent_samples, out_file, fps=10):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.latent_samples = latent_samples
        
        self.writer = imageio.get_writer(out_file, fps=fps)
    
    def generate_and_save(self, gen, enc):
        reconstructed_train_samples = gen(enc(self.train_samples, training=False), training=False)
        reconstructed_test_samples = gen(enc(self.test_samples, training=False), training=False)
        generated_samples = gen(self.latent_samples, training=False)
        
        rows = [
            self.train_samples,
            reconstructed_train_samples,
            self.test_samples,
            reconstructed_test_samples,
            generated_samples
        ]
        
        # concatenate images horizontally in rows
        rows = [K.concatenate(tf.unstack(row, axis=0), axis=1) for row in rows]
        # concatenate rows vertically in a single image
        image = K.concatenate(rows, axis=0).numpy()
        
        # convert [-1, 1] to [0, 255]
        image = ((image + 1) * 127.5).astype(np.uint8)
        self.writer.append_data(image)
        
        # save last image to inspect it during training
        imageio.imwrite('preview.png', image)
    
    def close(self):
        self.writer.close()
        
        


