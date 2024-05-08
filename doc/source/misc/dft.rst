Discrete Fourier transform (DFT)
================================



**Discrete Fourier transform (DFT)** 

The discrete Fourier transform (DFT) is the digital version of Fourier transform, which is used to analyze digital signals. The formula of DFT is:

:math:`X(k)=\sum_{n=0}^{N-1} x(n)e^{-2 \pi i k n/N}`

DFT incurs a complexity of :math:`O(N^2)`. 

A naive Python program can be easily done. Save the following python code as *dft_test.py* (However, the efficiency is not satisfactory. Python has provided an FFT which is faster than naive DFT.)


.. literalinclude::  ../codes/dft_test.py

Now run *dft_test.py* program in command line::
   
    $ python3 dft_test.py
    $ Is DFT close to fft? True
    

**Inverse Discrete Fourier transform (IDFT)**

Inverse discrete Fourier transform (IDFT)

:math:`x(n)= \frac{1}{N}\sum_{k=0}^{N-1} X(k)e^{2 \pi i k n/N}` 
 


Now test a function of IDFT *idft_test.py*

.. literalinclude::  ../codes/idft_test.py

Now run *idft_test.py* program in command line::
   
    $ python3 idft_test.py
    $ Is IDFT close to original? True
    
