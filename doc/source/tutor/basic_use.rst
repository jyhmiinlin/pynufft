Basic use of PyNUFFT
====================

This section navigates you through the basic use of PyNUFFT. 

---------------------------
Initiating a PyNUFFT object
---------------------------

We can initiate a PyNUFFT by importing the NUFFT object::

   # import NUFFT class
   from pynufft import NUFFT
   
   # Initiate the NufftObj object
   NufftObj = NUFFT()


The NufftObj object has been created but at this point it is still empty.

Now we have to plan the NufftObj by calling the plan() method. 
The plan() method takes the input variables and plans for the object. 
Now we can plan for the NufftObj object given the non-Cartesian coordinates (om).
 

In the following code we have 100 random samples spreading across the 2D plane.  

 ::

   # generating 2D random coordinates
   import numpy
   om = numpy.random.randn(100, 2)

 
-------------------------
Plan for the NUFFT object
-------------------------

Now we call: ::

   NufftObj.plan(om, Nd, Kd, Jd)
   

See :py:class: `pynufft.NUFFT`


-------------
Forward NUFFT
-------------
   
The forward NUFFT transforms the image into non-Cartesian samples. ::

   y = NufftObj.forward(image)
   
The image.shape is equal to Nd. The returned y has a shape which is equal to (M, )
   
See :py:func:`pynufft.NUFFT.forward`

-------------
Adjoint NUFFT
-------------

The adjoint NUFFT transforms the non-Cartesian samples into the image ::

   x2 = NufftObj.adjoint(y)
   
y has a shape which is equal to (M, ). The returned image.shape is equal to Nd. Note that the result is divided by a scaling factor numpy.prod(NufftObj.Kd), which is equal to the default normalization factor of numpy.fft.ifft.






