# -*- coding: utf-8 -*-
"""
@author: Guojian Wang
"""

import os
import re
from setuptools import setup, find_packages


def read(filename):
    f = open(filename)
    r = f.read()
    f.close()
    return r

ver = re.compile("__version__ = \"(.*?)\"")
#m = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "refann", "__init__.py"))
#m = read(os.path.join(os.path.dirname(__file__), "refann", "__init__.py"))
m = read(os.path.join(os.getcwd(), "colfi", "__init__.py"))
version = ver.findall(m)[0]



setup(
    name = "colfi",
    version = version,
    keywords = ("pip", "ANN", "MDN", "MNN"),
    description = "Cosmological Likelihood-Free Inference with Neural Density Estimators",
    long_description = "",
    license = "MIT",

    url = "",
    author = "Guojian Wang",
    author_email = "gjwang2018@gmail.com",

    # packages = find_packages(),
    packages = ["colfi", "examples", "examples/linear", "examples/SNe_BAO", "examples/pantheon"],
    include_package_data = True,
    data_files = ["examples/data/Pantheon_SNe_NoName.txt",
                  "examples/data/Pantheon_Systematic_Error_Matrix.npy",
                  "examples/data/chain_pantheon_fwCDM_3params_cov.npy",
                  "examples/data/chain_pantheon_fwCDM_3params.npy",
                   ],
    platforms = "any",
    install_requires = ["coplot>=0.1.3", "smt>=1.0.0", "pandas>=1.1.3", "numpy>=1.19.5", 
                        "scipy>=1.6.2", "matplotlib>=3.4.1"]
)

