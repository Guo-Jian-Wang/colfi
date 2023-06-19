.. _installation:

Installation
************

Since colfi is a pure python module, it is easy to install.


Dependencies
============

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


Package managers
================

You can install colfi by using pip::

    $ sudo pip install colfi

or from source::

    $ git clone https://github.com/Guo-Jian-Wang/colfi.git    
    $ cd colfi
    $ sudo python setup.py install


.. how to use conda?


Test the installation
=====================

To test the correctness of the installation, you just need to download the `examples <https://github.com/Guo-Jian-Wang/colfi/tree/master/examples/linear>`_ and execute it in the examples directory by using the following command::

    $ python train_linear.py

