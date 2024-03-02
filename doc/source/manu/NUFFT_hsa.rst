NUFFT(device)
=============

**The NUFFT(device) class**

NUFFT(device) computes NUFFT on the CUDA/OpenCL device.

Defining the equispaced to non-Cartesian transform as  operator :math:`A`, the
NUFFT(device) class provides the following methods:

- forward() method computes the single forward operation :math:`A`.

- adjoint() method computes the scaled adjoint operation  :math:`A^H`. The result is divided by the scaling factor of numpy.prod(Kd).

- selfadjoint() method computes the single selfadjoint operation :math:`A^H A`.

- solve() method now only offers 'cg', 'dc', 'L1TVOLS'

**Attributes**


- NUFFT.ndims: the dimension

- NUFFT.ft_axes: the axes where the FFT takes place

- NUFFT.Nd: Tuple, the dimensions of the image

- NUFFT.Kd: Tuple, the dimensions of oversampled k-space

**Acceleration on PyCUDA/PyOpenCL**

The NUFFT_hsa was designed to accelerate the NUFFT function 
on the multi-core CPU and GPU, using PyOpenCL and PyCUDA backends.
This was made possible by using Reikna meta-package. 

If multiple NUFFT(device) objects are created with the PyCUDA backend, 
each call can be executed only after the  context has 'popped up'. This is 
achieved by the decorator function push_cuda_context():  
calling NUFFT(device) methods will trigger the decorator and get the context popped up. 
However, PyOpenCL has no such restriction 
and the call will automatically bypass the decorator for the NUFFT(device) 
with the PyOpenCL backend. 

Different objects can be constructed on different PyCUDA and PyOpenCL backends. 

**The life-cycle of the PyNUFFT(device) object**


NUFFT(device) employs the plan-execution two-stage model.
This can maximize the runtime speed, at the cost of the extra precomputation times and extra memory.

Instantiating an NUFFT(device) instance also initiates the context and defines some instance attributes. 
The context is linked to the accelerator and the kernels are compiled on the chosen device.
Instance attributes will be replaced later when the method plan() is called.


Then the plan() method calls the helper.plan() function, 
which constructs the scaling factor and the interpolator.  
The interpolator is precomputed and stored as multiple 1D ELLpack (ELL) sparse matrices. 
Each ELL matrix preserves the sparse matrix as the data and column indices. 
Multi-dimensional interpolators are implemented as a concatenated multiple 1D ELL sparse matrices.
The actual data and column indices are inferred from the meshindex.
At the end of the plan() method, the offload() method transfers the 
precomputed arrays to the accelerator. 

The run-time computations reuse the saved scaling factors and 
interpolators.  

 
  
