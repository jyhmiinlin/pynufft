FFTW support (Compile Numpy with FFTW in Linux)
===============================================

The original Numpy source includes fftpack as its FFT implementation.

However, FFTW is unofficially supported in Numpy. The trick is to compile Numpy with a configuration file ~/.numpy-site.cfg. 

Create a Numpy configuration file in user's home directory::

    $ echo "[fftw]" >  ~/.numpy-site.cfg
    $ echo "libraries = fftw3" >>  ~/.numpy-site.cfg

Then  install Numpy from pip::

   $ pip uninstall numpy
   $ pip install numpy

