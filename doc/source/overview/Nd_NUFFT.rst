Multi-dimensional NUFFT
=======================
Multi-dimensional transforms are supported by PyNUFFT. 

The dimensionality of an imaging reconstruction problem is revealed as 
the number of axes of the Nd (or Kd, Jd) tuples.

For example, Nd = (256,256) indicates a 2D imaging reconstruction problem, 
in which the sizes of the x-y axes are 256 and 256, respectively. 

Normally, the matrix size of the k-space is twice the size of Nd. For example, 
Kd = (512,512) is appropriate for the above Nd = (256,256) problem. 

In batch mode, the 'batch' argument controls the number of channels. 
This will not affect the dimensionality of the image reconstruction problem.  
The batch model will be detailed in the 'batched NUFFT' section. 
 
:numref:`configuration_nufft` illustrates the variables for 1D, 2D, 3D NUFFT.




.. _configuration_nufft:

.. figure:: ../figure/configuration_nufft.png
   :width: 60%
   
   Configuration of 1D, 2D, and 3D NUFFT. 
   (A) 1D NUFFT: om is a numpy.array of the shape (M,1). 
   M is the number of non-Cartesian points. 
   Nd = (8, ) is the image domain grid size and Kd = (16, ) is the oversampled grid size. 
   Jd = (6, ) is the interpolator size.
   (B) 2D NUFFT: om is a numpy.array of the shape (M,2). 
   M is the number of non-Cartesian points. 
   Nd = (8, 8 ) is the image domain grid size and Kd = (16, 16 ) is the oversampled grid size. 
   Jd = (6, 6 ) is the interpolator size.   
   (C) 3D NUFFT: om is a numpy.array of the shape (M,3). 
   M is the number of non-Cartesian points. 
   Nd = (8, 8, 8 ) is the image domain grid size and Kd = (16, 16, 16 ) is the oversampled grid size. 
   Jd = (6, 6, 6 ) is the interpolator size.      



   
