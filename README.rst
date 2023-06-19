CoLFI
=====

**CoLFI (Cosmological Likelihood-Free Inference with Neural Density Estimators)**

CoLFI is a framework to estimate cosmological parameters based on neural density estimators. It is an alternative to the traditional `Markov chain Monte Carlo (MCMC) <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ method and has advantages over MCMC.

CoLFI can be applied to the research of cosmology and even other broader scientific fields.

It is proposed by `Guo-Jian Wang, Cheng Cheng, Yin-Zhe Ma, et al. (2023) <>`_.


Documentation
-------------

The documentation can be found at `colfi.readthedocs.io <https://colfi.readthedocs.io>`_.


Attribution
-----------

If you use this code in your research, please cite `Guo-Jian Wang, Cheng Cheng, Yin-Zhe Ma, et al., "CoLFI: Cosmological Likelihood-free Inference with Neural Density Estimators", ApJS, XX, XX, (2023) <>`_.

If you use the MDN method of this code, please also cite `Guo-Jian Wang, Cheng Cheng, Yin-Zhe Ma, Jun-Qing Xia, "Likelihood-free Inference with the Mixture Density Network", ApJS, 262, 24 (2022) <https://doi.org/10.3847/1538-4365/ac7da1>`_.

If you use the ANN method of this code, please also cite `Guo-Jian Wang, Si-Yao Li, Jun-Qing Xia, "ECoPANN: A Framework for Estimating Cosmological Parameters Using Artificial Neural Networks", ApJS, 249, 25 (2020) <https://doi.org/10.3847/1538-4365/aba190>`_.


Dependencies
------------

The main dependencies of colfi are (need to install manually):

* `PyTorch <https://pytorch.org/>`_
* `CUDA <https://developer.nvidia.com/cuda-downloads>`_ (optional, but suggested)

and some commonly used modules (will be installed automatically):

* `coplot <https://github.com/Guo-Jian-Wang/coplot>`_
* smt
* numpy
* scipy
* pandas
* matplotlib


Installation
------------

You can install colfi by using pip::

    $ sudo pip install colfi

or from source::

    $ git clone https://github.com/Guo-Jian-Wang/colfi.git    
    $ cd colfi
    $ sudo python setup.py install


License
-------

Copyright 2022-2023 Guojian Wang

colfi is free software made available under the MIT License. For details see the LICENSE file.
