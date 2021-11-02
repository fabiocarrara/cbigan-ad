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
**Combining GANs and AutoEncoders for Efficient Anomaly Detection**.
_Fabio Carrara, Giuseppe Amato, Luca Brombin, Fabrizio Falchi, Claudio Gennaro._
In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 3939-3946). IEEE.
[[arXiv](https://arxiv.org/abs/2011.08102), [DOI](https://doi.org/10.1109/ICPR48806.2021.9412253)]

    @inproceedings{carrara2021combining,
      title={Combining gans and autoencoders for efficient anomaly detection},
      author={Carrara, Fabio and Amato, Giuseppe and Brombin, Luca and Falchi, Fabrizio and Gennaro, Claudio},
      booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
      pages={3939--3946},
      year={2021},
      organization={IEEE}
    }
