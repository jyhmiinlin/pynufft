CPU and GPU (HSA)
=================

The PyNUFFT ran originally on Numpy/Scipy. Unfortunately the default Numpy/Scipy is most efficient on a single CPU core. 

Later it was ported to PyCUDA and PyOpenCL, which allows us to leverage the speed of multi-core CPU and GPU.   

Mixing NUFFTs with CPU/GPU is possible but has no warranty. 

The class methods are listed in :numref:`dimension_table`

.. _dimension_table:
.. list-table:: Methods implemented in NUFFT
   :widths: 25 12 12 30
   :header-rows: 1

   * - method name
     - NUFFT()
     - NUFFT(helper.device_list[0])
     - References
   * - __init__()
     - ✓
     - ✓
     - Constructor
   * - plan()
     - ✓
     - ✓
     - Planning the instance
   * - forward()
     - ✓ 
     - ✓
     - Forward NUFFT :math:`A`
   * - adjoint()
     - ✓
     - ✓
     - Adjoint NUFFT :math:`A^H`
   * - offload()
     - ×          
     - ✓
     - Offload the NUFFT_hsa() to device. 
   * - x2xx()
     - ✓          
     - ✓
     - Apply the scaling factor 
   * - xx2k()
     - ✓          
     - ✓
     - Oversampled FFT    
   * - k2y()
     - ✓          
     - ✓
     - Interpolation
   * - k2vec()
     - ✓          
     - ×   
     - Reshape the k-space to the vector       
   * - vec2y()
     - ✓          
     - ×   
     - Multiply the vector to generate the data          
   * - vec2k()
     - ✓          
     - ×   
     - Reshape the vector to k-space      
   * - y2vec()
     - ✓          
     - ×   
     -  Multiply the data to get the vector       
   * - y2k()
     - ✓          
     - ✓
     - Adjoint of k2y()
   * - k2xx()
     - ✓          
     - ✓
     - Inverse FFT (excessive parts are cropped)
   * - xx2x()
     - ✓          
     - ✓
     - Apply the scaling factor      
   * - _precompute
     - ✓          
     - ✓
     - Apply the scaling factor                   

     
---------------------
Parameters of PyNUFFT
---------------------


Below we summarize the required variables in :numref:`parameter_table`


.. _parameter_table:
.. list-table:: Parameters of the plan() method
   :widths: 25 12 12 30
   :header-rows: 1

   * - Parameter
     - NUFFT
     - NUFFT(helper.device_list()[0])
     - References
   * - om (Numpy Array)
     - ✓
     - ✓  
     - Non-Cartesian coordinates (M, dim)
   * - Nd (tuple)
     - ✓
     - ✓ 
     - Size of the image grid
   * - Kd (tuple)
     - ✓ 
     - ✓ 
     - Size of the oversampled Fourier grid
   * - Jd (tuple)
     - ✓ 
     - ✓
     - Size of the interpolator
   * - ft_axes (tuple)
     - optional 
     - optional
     - FFT on the given axes (default = None (all axes))    
   * - radix (int)
     - ×
     - optional
     - radix (default = 1)         