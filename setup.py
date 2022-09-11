from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Operating System :: Microsoft :: Windows :: Windows 11',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Private :: Do Not Upload',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering :: Chemistry'
]

setup(
    name='PyChrom',
    version=0.0.1,
    description='Module to provide tools to process and analyse chromatographic data from different sources such as UPLC, LCMS, GCMS, etc.',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url=''
)