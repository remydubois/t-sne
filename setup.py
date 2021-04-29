from setuptools import find_packages, setup

setup(
    name="tsne",
    version="0.1.0",
    description="Numba-base parallelized t-Stochastic Neighbors Embedding",
    author="Rémy Dubois",
    install_requires=["numpy==1.19.5", "numba==0.53.1", "setuptools"],
    url="https://github.com/remydubois/tsne",
    keywords=[],
    classifiers=["Programming Language :: Python :: 3", "Topic :: Scientific/Engineering"],
    packages=find_packages(exclude=("tests")),
)
