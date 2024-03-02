NUFFT
=====

**The NUFFT class**

The NUFFT class encapsulates the NUFFT function using the Numpy/Scipy environment. 
This allows portability so the NUFFT() can be easily ported to Windows and Linux.
Users can install their favourite Python distribution. 
So far, I have tested Anaconda, intel-python, intel-numpy and open-source python.

However, the performance of NUFFT is therefore impacted by the Numpy implementation.  


Defining the equispaced to non-Cartesian transform as  operator :math:`A`, the
NUFFT class provides the following methods:

- forward() method computes the single forward operation :math:`A`.

- adjoint() method computes the scaled adjoint operation  :math:`A^H`. The result is divided by a scaling factor of numpy.prod(Kd).

- selfadjoint() method computes the single selfadjoint operation :math:`A^H A`.


- solve() method links many solvers in pynufft.linalg.solver,
          which is based on the solvers of scipy.sparse.linalg.cg,
          scipy.sparse.linalg. 'lsqr', 'dc', 'bicg', 'bicgstab', 'cg',
          'gmres', 'lgmres'


**Attributes**

- NUFFT.ndims: the dimension

- NUFFT.ft_axes: the axes where the FFT takes place

- NUFFT.Nd: Tuple, the dimensions of the image

- NUFFT.Kd: Tuple, the dimensions of the oversampled k-space


**The life-cycle of an NUFFT object**


NUFFT employs the plan-execution two-stage model.
This can maximize the runtime speed, at the cost of the extra precomputation times and extra memory.


Instantiating an NUFFT instance defines only some instance attributes. Instance attributes will be replaced later once the method plan() is called.
  
Then the plan() method will call the helper.plan() function, 
which constructs the scaling factor and the interpolator.  
The interpolator is precomputed and stored in the Compressed Sparse Row (CSR) format. 
See `scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ and 
`CSR in Wikipedia <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_.   
  
Once the object has been planned, the forward() and adjoint() methods reuse the saved scaling factors and interpolators. 

