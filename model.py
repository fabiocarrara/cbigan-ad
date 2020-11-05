import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model

from functools import partial

init = dict() # kernel_initializer='he_normal')
common = dict(padding='same', **init)
bn_layers = {
    'batch': L.BatchNormalization,
    'layer': L.LayerNormalization,
    'instance': tfa.layers.InstanceNormalization,
    'none': lambda: (lambda x: x)  # complex way to say no-op
}

def get_bn_layer(bn='none'):
    assert bn in bn_layers, f"Unsupported normalization layer {bn}"
    return bn_layers[bn]

# L.Layer implements identity by default
def g_block(x, n_filters, upsample=True, upsample_type='bilinear', use_bias=True, bn=L.Layer, act=L.ReLU):
    if upsample:
        if upsample_type == 'bilinear':
            x = L.UpSampling2D(interpolation='bilinear')(x)
        else:
            x = L.Conv2DTranspose(n_filters, kernel_size=1, strides=2, **common)(x)
    
    skip = L.Conv2D(n_filters, kernel_size=1, use_bias=use_bias, **common)(x)
    skip = bn()(skip)
    
    x = L.Conv2D(n_filters, kernel_size=3, use_bias=use_bias, **common)(x)
    x = bn()(x)
    x = act()(x)
    x = L.Conv2D(n_filters, kernel_size=3, use_bias=use_bias, **common)(x)
    x = bn()(x)
    x = act()(x)
    x = L.Conv2D(n_filters, kernel_size=1, **common)(x)

    x = L.Add()([x, skip])
    x = bn()(x)
    x = act()(x)

    return x


def d_block(x, n_filters, pool=True, use_bias=True, bn=L.Layer, act=L.ReLU):
    
    skip = L.Conv2D(n_filters, kernel_size=1, use_bias=use_bias, **common)(x)
    skip = bn()(skip)
    
    x = L.Conv2D(n_filters, kernel_size=3, use_bias=use_bias, **common)(x)
    x = bn()(x)
    x = act()(x)
    x = L.Conv2D(n_filters, kernel_size=3, use_bias=use_bias, **common)(x)
    x = bn()(x)
    x = act()(x)
    x = L.Conv2D(n_filters, kernel_size=1, **common)(x)

    x = L.Add()([x, skip])
    x = bn()(x)
    x = act()(x)

    if pool:
        x = L.AveragePooling2D()(x)

    return x
    

def make_generator(latent_size, channels=3, upsample_first=True, upsample_type='bilinear', bn='none', act='lrelu'):
    use_bias = not bn
    bn = get_bn_layer(bn)
    act = partial(L.LeakyReLU, alpha=0.2) if act == 'lrelu' else L.ReLU
    g_common = dict(upsample_type=upsample_type, use_bias=use_bias, bn=bn, act=act)
    
    i = L.Input(shape=[latent_size])
    x = L.Dense(2 * 2 * 16 * channels, use_bias=use_bias, **init)(i)
    x = bn()(x)
    x = L.Reshape([2, 2, 16 * channels])(x) # 2 x 2
    
    # next output is 4x4 for objects, but remains 2x2 for textures
    # therefore, final output is 128x128 for objects, and 64x64 for textures
    x = g_block(x, 16 * channels, upsample=upsample_first, **g_common)  # 4 x 4
    x = g_block(x,  8 * channels, **g_common)  # 8 x 8
    x = g_block(x,  4 * channels, **g_common)  # 16 x 16
    x = g_block(x,  3 * channels, **g_common)  # 32 x 32
    x = g_block(x,  2 * channels, **g_common)  # 64 x 64
    x = g_block(x,  1 * channels, **g_common)  # 128 x 128
    
    o = L.Conv2D(filters=3, kernel_size=1, activation='tanh', **common)(x)
    
    return Model(inputs=i, outputs=o, name='generator')


def make_encoder(image_size, latent_size, channels=3, bn='none', act='lrelu'):
    use_bias = not bn
    bn = get_bn_layer(bn)
    act = partial(L.LeakyReLU, alpha=0.2) if act == 'lrelu' else L.ReLU
    e_common = dict(use_bias=use_bias, bn=bn, act=act)
    
    i = L.Input(shape=[image_size, image_size, 3])

    x = d_block(i,  1 * channels, **e_common)  # 64 x 64
    x = d_block(x,  2 * channels, **e_common)  # 32 x 32
    x = d_block(x,  3 * channels, **e_common)  # 16 x 16
    x = d_block(x,  4 * channels, **e_common)  # 8 x 8
    x = d_block(x,  8 * channels, **e_common)  # 4 x 4
    x = d_block(x, 16 * channels, pool=False, **e_common)  # 4 x 4
    x = L.Flatten()(x)

    x = L.Dense(16 * channels, use_bias=use_bias, **init)(x)
    x = bn()(x)
    x = act()(x)
    o = L.Dense(latent_size, **init)(x)

    return Model(inputs=i, outputs=o, name='encoder')


def make_discriminator(image_size, latent_size, channels=3, bn='none', act='lrelu'):
    use_bias = not bn
    bn = get_bn_layer(bn)
    act = partial(L.LeakyReLU, alpha=0.2) if act == 'lrelu' else L.ReLU
    d_common = dict(use_bias=use_bias, bn=bn, act=act)
    
    ii = L.Input(shape=[image_size, image_size, 3]) 
    il = L.Input(shape=[latent_size])
    
    # latent path
    l = L.Dense(512, use_bias=use_bias, **init)(il)
    l = bn()(l)
    l = act()(l)
    l = L.Dense(512, use_bias=use_bias, **init)(l)
    l = bn()(l)
    l = act()(l)
    l = L.Dense(512, use_bias=use_bias, **init)(l)
    l = bn()(l)
    l = act()(l)
    
    # image path
    x = d_block(ii, 1 * channels, **d_common)  # 64 x 64
    x = d_block(x,  2 * channels, **d_common)  # 32 x 32
    x = d_block(x,  3 * channels, **d_common)  # 16 x 16
    x = d_block(x,  4 * channels, **d_common)  # 8 x 8
    x = d_block(x,  8 * channels, **d_common)  # 4 x 4
    x = d_block(x, 16 * channels, pool=False, **d_common)  # 4 x 4
    x = L.Flatten()(x)
    
    # common path
    x = L.Concatenate()([x, l])
    x = L.Dense(16 * channels, use_bias=use_bias, **init)(x)
    x = bn()(x)
    x = act()(x)
    x = L.Dense(1, **init)(x)

    return Model(inputs=[ii, il], outputs=x, name='discriminator')
    

if __name__ == '__main__':
    g = make_generator(64, 64, upsample_first=False, bn='batch')
    e = make_encoder(64, 64, bn='instance')
    d = make_discriminator(64, 64, bn='layer')
    
    g.summary()
    e.summary()
    d.summary()
    
