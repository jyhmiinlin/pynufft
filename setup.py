
from setuptools import setup

import sys
#if sys.version_info[0] == 2:
#    sys.exit("Sorry, Python 2 is not supported yet")
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pynufft',
      version='2024.1.2',
      description='Python non-uniform fast Fourier transform (PyNUFFT)',
      author='PyNUFFT',
      author_email='pynufft@gmail.com',
      url = 'https://github.com/jyhmiinlin/pynufft', # use the URL to the github repo
      install_requires = ['numpy', 'scipy'],
      license='The MIT License',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['pynufft'],
      package_dir={'pynufft':'.'},
      package_data={'pynufft':['nufft','src','tests','data','linalg','nufft/*','src/*','src/*/*','tests/*','example/*', 'linalg/*']},
      include_package_data=True,
      zip_safe=False)
	
