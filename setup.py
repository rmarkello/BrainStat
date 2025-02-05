#!/usr/bin/env python3

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

from os import path
import setuptools


here = path.abspath(path.dirname(__file__))


with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brainstat",
    version="0.0.3",
    author="MNI-MICA Lab and MPI-CNG Lab",
    author_email="reinder.vosdewael@gmail.com, sheymaba@gmail.com",
    description="A toolbox for statistical analysis of neuroimaging data",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/MICA-LAB/BrainStat",
    packages=setuptools.find_packages(),
    license="BSD",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6",
    test_require=["pytest", "gitpython"],
    install_requires=[
        "abagen",
        "brainspace>=0.1.1",
        "neurosynth",
        "nibabel",
        "nilearn>=0.7.0",
        "nimare",
        "numpy>=1.16.5",
        "numpy_groupies",
        "pandas",
        "scikit_learn",
        "scipy>=1.3.3",
        "trimesh",
    ],
    project_urls={  # Optional
        "Documentation": "https://brainstat.readthedocs.io",
        "Bug Reports": "https://github.com/MICA-LAB/BrainStat/issues",
        "Source": "https://github.com/MICA-LAB/BrainStat/",
    },
)
