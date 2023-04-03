from setuptools import setup, find_packages


__version__ = "0.0.6"
__author__ = "Rodrigo Silva Nascimento Mancini"


classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Operating System :: Microsoft :: Windows :: Windows 11",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Chemistry",
]

setup(
    name="PyChrom",
    version=__version__,
    description="Module to provide tools to process and analyse chromatographic data from different sources such as UPLC, LCMS, GCMS, etc.",
    long_description=open("README.txt").read() + "\n\n" + open("CHANGELOG.txt").read(),
    url="https://github.com/rdgsnm/PyChrom",
    author=__author__,
    author_email="rdgsnm@gmail.com",
    license="MIT",
    classifiers=classifiers,
    keywords="chromatography, chemometrics, data preprocessing",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "sklearn"],
)
