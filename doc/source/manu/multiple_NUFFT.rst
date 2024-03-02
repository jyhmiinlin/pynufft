Multiple NUFFT instances
========================

Multiple NUFFT instances (of mixed types) can coexist at the same time.
Each instance has its own memory. 
However,  multiple instances  may be penalized by hardware and reduced runtime speed.

Multiple instances can be planned after all the instances have been created.
Alternatively, each instance can be planned immediately after being created.   

Note that multiple PyCUDA instances were made possible since 2019.1.1, 
by introducing the push_cuda_context() decorating function. 
Versions earlier than 2019.1.1 do not support multiple PyCUDA backends.
This is because every call to a CUDA context will require the current 
context to pop up to the top of the stack of the contexts.

PyOpenCL does not have the context pop-up issue but please always use the newest version. 


**Multiple NUFFT instances**

Multiple instances can be planned after all the instances have been created::

   # Create the first NUFFT
   NufftObj1 = NUFFT()
   # Create the second NUFFT
   NufftObj2 = NUFFT()
   # Plan the first instance
   NufftObj1.plan(om1, Nd, Kd, Jd)
   NufftObj2.plan(om2, Nd, Kd, Jd)
   
   
or each instance can be planned once it has been created::

   # Create the first NUFFT
   NufftObj1 = NUFFT()
   NufftObj1.plan(om1, Nd, Kd, Jd)
   
   # Create the second NUFFT
   NufftObj2 = NUFFT()
   NufftObj2.plan(om2, Nd, Kd, Jd)  
   
   y1 = NufftObj1.forward(x)
   y2 = NufftObj2.forward(x)    

**Multiple NUFFT_hsa instances**

Like NUFFT, each instance can be planned immediately after being created::


   from pynufft import NUFFT, helper
   # Create the first NUFFT(device)
   NufftObj1 = NUFFT(helper.device_list()[0])
   NufftObj1.plan(om1, Nd, Kd, Jd)
   
   # Create the second NUFFT(device)
   NufftObj2 = NUFFT(helper.device_list()[0])
   NufftObj2.plan(om2, Nd, Kd, Jd)
   
   y1 = NufftObj1.forward(x)
   y2 = NufftObj2.forward(x)


Multiprocessing (experimental)
==============================


The multiprocessing module is the built-in parallel method of C-Python. 
PyNUFFT may (experimentally) work together with the multiprocessing 
module of Python.  

The mutliprocessing module relies on pickle to serialize the data, whereas 
the PyCUDA and PyOpenCL contexts are "unpicklable". 
Thus, I found that multiprocessing for PyNUFFT must fulfil the following conditions: (1)
each NUFFT instance should be created and then executed in a separate process; 
(2) any CUDA/PyOpenCL related object cannot be sent or planned in advance, and
(3) taskset should be used to assign a process to a specified CPU core. 

It is the user's responsibility to take care of the hardware (total memory and IO). 



One example of working with multiprocessing (mixed CUDA and OpenCL backends) is as follows.
In this example, an "atomic_NUFFT" class is created as a high-level wrapper for the creation and execution of NUFFT.
This example has only been tested in Linux because parallel computing is highly platform dependent.

.. literalinclude::  ../../../tests/parallel_NUFFT.py
