The Python Non-uniform fast Fourier transform (PyNUFFT)
=======================================================



**Purpose**

The PyNUFFT user manual documents the *Python non-uniform fast Fourier transform*, a Python package for non-uniform fast Fourier transform.

PyNUFFT was created for practical purposes in industry and in research. 

If you find PyNUFFT useful, please cite:

*Lin, Jyh-Miin. "Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU)." Journal of Imaging 4.3 (2018): 51.*

and

*J.-M. Lin and H.-W. Chung, Pynufft: python non-uniform fast Fourier transform for MRI Building Bridges in Medical Sciences 2017, St John's College, CB2 1TP Cambridge, UK*

Users of PyNUFFT should be familiar with discrete Fourier transform (DFT). 


**The min-max interpolator**

- PyNUFFT translates the min-max interpolator to Python. The min-max interpolator is described in the literature:

*Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.*

 
**Current status of PyNUFFT**

- The current PyNUFFT offers NUFFT() or Reikan/PyCUDA/PyOpenCL (NUFFT(device), device is in helper.device_list()). 

Switch between CPU and GPU by selecting device = pynufft.helper.device_list[0] (0 is the first device in the system) 

- LGPLv3 and AGPL (for web service)
 
 .. figure:: ../figure/speed_accuracy_comparisons.png
   :width: 100 %

