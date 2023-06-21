CoLFI
=====

**CoLFI** (``Co``\smological ``L``\ikelihood-``F``\ree ``I``\nference) is a framework to estimate cosmological parameters based on neural density estimators (ANN, MDN, and MNN) proposed by `Guo-Jian Wang, Cheng Cheng, Yin-Zhe Ma, et al. (2023) <https://arxiv.org/abs/2306.11102>`_.

It is an alternative to the traditional `Markov chain Monte Carlo (MCMC) <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ method and has advantages over MCMC.

As a general method of parameter estimation, CoLFI can be used for research in many scientific fields. The code colfi is available for free from `GitHub <https://github.com/Guo-Jian-Wang/colfi>`_. It can be executed on GPUs or CPUs.


.. image:: https://img.shields.io/badge/GitHub-colfi-blue.svg?style=flat
    :target: https://github.com/Guo-Jian-Wang/colfi
.. image:: https://img.shields.io/badge/License-MIT-green.svg?style=flat
    :target: https://github.com/Guo-Jian-Wang/colfi/blob/master/LICENSE
.. image:: https://img.shields.io/badge/ApJS-CoLFI-blue.svg?style=flat
    :target: https://arxiv.org/abs/2306.11102
.. image:: https://img.shields.io/badge/arXiv-2306.11102-gold.svg?style=flat
    :target: https://arxiv.org/abs/2306.11102


Attribution
===========

If you use this code in your research, please cite `Guo-Jian Wang, Cheng Cheng, Yin-Zhe Ma, et al., "CoLFI: Cosmological Likelihood-free Inference with Neural Density Estimators", ApJS, XX, XX, (2023) <https://arxiv.org/abs/2306.11102>`_.

If you use the MDN method of this code, please also cite `Guo-Jian Wang, Cheng Cheng, Yin-Zhe Ma, Jun-Qing Xia, "Likelihood-free Inference with the Mixture Density Network", ApJS, 262, 24 (2022) <https://doi.org/10.3847/1538-4365/ac7da1>`_.

If you use the ANN method of this code, please also cite `Guo-Jian Wang, Si-Yao Li, Jun-Qing Xia, "ECoPANN: A Framework for Estimating Cosmological Parameters Using Artificial Neural Networks", ApJS, 249, 25 (2020) <https://doi.org/10.3847/1538-4365/aba190>`_.

..
	If you use this code in your research, please cite our paper
	(`ApJS <>`_,
	`arXiv <>`_,
	`ADS <>`_,
	`BibTex <>`_).
	
	If you use the ANN method of this code in your research, please also cite the ANN paper
	(`ApJS <https://doi.org/10.3847/1538-4365/aba190>`_,
	`arXiv <https://arxiv.org/abs/2005.07089>`_,
	`ADS <https://ui.adsabs.harvard.edu/abs/2020ApJS..249...25W/abstract>`_,
	`BibTex <https://ui.adsabs.harvard.edu/abs/2020ApJS..249...25W/exportcitation>`_).
	
	If you use the MDN method of this code in your research, please also cite the MDN paper
	(`ApJS <https://doi.org/10.3847/1538-4365/ac7da1>`_,
	`arXiv <https://arxiv.org/abs/2207.00185>`_,
	`ADS <https://ui.adsabs.harvard.edu/abs/2022ApJS..262...24W/abstract>`_,
	`BibTex <https://ui.adsabs.harvard.edu/abs/2022ApJS..262...24W/exportcitation>`_).


How to use CoLFI
================

First, you are probably going to needs to see the :ref:`introduction` guide to learn the basic principles of CoLFI. After that, you may need to install colfi on your computer according to the :ref:`installation` guide, and then following the :ref:`quickStart` guide to learn how to use it. If you need more detailed information about a specific function, the :ref:`package_reference` below should have what you need.


Contents:
=========

.. toctree::
   :maxdepth: 2
   
   introduction
   installation
   quickStart
   package_reference
   release_history


License
=======

Copyright 2022-2023 Guojian Wang

colfi is free software made available under the MIT License. For details see the `LICENSE <https://github.com/Guo-Jian-Wang/colfi/blob/master/LICENSE>`_.

