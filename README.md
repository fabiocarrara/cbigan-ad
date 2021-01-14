# Consistency Bidirectional GAN (CBiGAN)

CBiGAN: a combined model that generalizes Bidirectional GANs (BiGANs) and AutoEncoders, applied to anomaly detection in images.
The repo provides training and evaluation code for the MVTecAD anomaly detection benchmark.

Also provides a TensorFlow2 implementation of BiGAN following the Wasserstein GAN (WGAN) formulation.

## Getting started

You need:
 - Python 3
 - Tensorflow 2.4.0
 - packages in requirements.txt

You can use the [Dockerfile](./Dockerfile) to build an image.

## Train on MVTec-AD

[Download the whole MVTec-AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract into `data/mvtec-ad`.

Check out the `train.py` script for training parameters:
```sh
python train.py -h
```


## Reference
**Combining GANs and AutoEncoders for Efficient Anomaly Detection** [[arXiv](https://arxiv.org/abs/2011.08102)]
Fabio Carrara, Giuseppe Amato, Luca Brombin, Fabrizio Falchi, Claudio Gennaro

    @article{carrara2020combining,
      title={Combining GANs and AutoEncoders for Efficient Anomaly Detection},
      author={Carrara, Fabio and Amato, Giuseppe and Brombin, Luca and Falchi, Fabrizio and Gennaro, Claudio},
      journal={arXiv preprint arXiv:2011.08102},
      year={2020}
    }
