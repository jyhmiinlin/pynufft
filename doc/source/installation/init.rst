Installation
============

-------------------
System requirements
-------------------

**CPU**

Each PyNUFFT instance is designed to be executed on a single node. 
PyNUFFT has no built-in distributed computing on multiple nodes, 
but the users can design their own one. 

For example, users may install PyNUFFT on multiple nodes and control them through the network.  

Multple NUFFT instances on a single node provided that the total memory is sufficient to keep all instances. 

We recommend one or more modern x86_64 processors on a single node. Successful stories include Intel® Core™ i7-6700HQ Processor, 
Intel® Xeon® W-2125, Intel® Core™ i9-7900X. 



**Memory**

A general instruction is that the memory should be sufficient for computing a single NUFFT object, which is dependent on the type of problem. 
  
A single 2D problem of 256 × 256 matrix can be computed on a system with 8GB memory. 
 
For 3D NUFFT, it is not uncommon for a single NUFFT object to consume more than 200GB memory.

**GPU**

Each PyNUFFT instance is initiated on a single GPU. An instance cannot be distributed across multiple GPUs. 
(PyNUFFT does not use cuFFT.)

However, multiple NUFFT_hsa instances may be initiated both on a single GPU and on multiple GPUs, 
but the performance may be impacted (limited by the memory, PCI-E bus or GPU cores).  

To use GPU, a recent NVIDIA's GPU (after Maxwell, Pascal) with the recent drivers should be working properly. 



The newest nvidia-driver versions of 510.54 or later is recommended. 
Earlier versions may work but please be informed that Nvidia may discontinue support for outdated drivers. 

A general rule is that the memory on the GPU has to be sufficient for computing the NUFFT problem.
Successful stories include NVIDIA Geforce GTX 965m/GTX 1070 maxQ/1060 6GB, 
NVIDIA Titan V100 (Amazon Web Services), 
NVIDIA Titan X Pascal, 
and NVIDIA Quadro P6000. 

**Operating System**

Ubunut 16.04 - 18.04 are recommended. 

Windows 10 has been tested but it requires Microsoft Studio 2015 community. Please refer to the following special topic about the installation under Windows 10. 

--------
Software
--------

**Python**
   
Users must be familiar with Python and its pip packaging system.  
Python 3.9 - 3.10 are currently supported (Python 2 has been discontinued). 

To run the NUFFT, the basic CPython, Numpy and Scipy packages must be available on the system.
IronPython is compatible with CPython so ipython might be useful. 

PyNUFFT can be installed through the pip command. 
Optionally, users can clone the github repository and build the package from the local folder. 

**Compiler**

NUFFT class does not require a compiler. 

However, NUFFT relies on the JIT (just-in-time) compilation mechanism of Reikna/PyCUDA/PyOpenCL. 
The supporting compiler may be:

- gcc-11.2.1

- Microsoft (R) Visual Studio 2022 community edition. 
(Please refer to the following section: special topic: Installation under Windows 10). 

To accelerate the code on the graphic processing unit (GPU), 
Reikna, PyCUDA, PyOpencl must be available. Please refer the following special topic: Installation of OpenCL.

CUDA programming skills are not strictly needed.  
However, it may be helpful if users understand GPU programming. 

--------------------
General Installation
--------------------

Continuum's Anaconda_ environment should provide all the above packages. 

.. _Anaconda: https://www.continuum.io/downloads

**Installation using pip**

Install pynufft by using the pip_ command::

   $ pip install pynufft

.. _pip: https://en.wikipedia.org/wiki/Pip_(package_manager)
    
**Installation from Git Repository**

git_ is a version control program, which allows you to clone the latest code base from the pynufft_ repository::
   
   $ git clone https://github.com/jyhmiinlin/pynufft
   $ cd pynufft
   $ python setup.py install --user 

.. _git: https://en.wikipedia.org/wiki/Git
.. _pynufft: https://github.com/jyhmiinlin/pynufft

**Uninstall pynufft**

Simply use "pip uninstall" to remove pynufft::

    $ pip uninstall pynufft


**Test whether the installation is successful (deprecated)**

- Ask pynufft@gmail.com for technical support.

--------------
Special topics
--------------

.. toctree::
   :maxdepth: 2
   
   Linux
   Windows
   Macbook
   OpenCL
   
