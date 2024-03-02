Version history
===============

**v2022.2.4rc1**

- Explain the scaling factor for NUFFT.adjoint in the document.

**v2022.2.3**

- Same as v2022.2.3rc2 

**v2022.2.3-rc2**

- Add experimental NUFFT_cupy for cupy backend.

- Change scipy.linalg.pinv2 (deprecate) to pinv. Track for possible errors.  

**v2022.2.3-rc1**

- Add experimental NUFFT_torch, NUFFT_tf

**v2022.2.2**

- Maintenance release

**v2022.2.1**

- Maintenance release

**v2022.1.1**

- some typos fixed.
- move to Python 3.10
- update installation for MSVC 2022 and Win 10

**v2021.2.0**

- typo fixed: ocl_add() imaglptr -> imagptr


**v2021.1.0**

- removed twosum algorithm.

**v2020.2.1**

- tested with an AMD card.
- Deprecate batch as this is not the most essential function in NUFFT.
- Deprecate NUFFT_cpu, NUFFT_hsa
- Update some docs.

**v2020.1.2**

- new_index: change the khatri_rao_k(), OMEGA_k() in helper.py; col += kindx[index];// + 1  
- Add code of conduct of Contributor Covenant v2.0
- Update some documentations


**v2020.1.0**

- add batch mode to nudft_cpu

**v2020.0.0**

- fix batch=1. This can cause error in Riekna fft.

**v2019.2.3**

-(experimental) Add the unified NUFFT() class. Now CPU, GPU, and legacy GPU mode are encapsuled in a single class. 

-Tested using Intel Neo OpenCL driver (see https://github.com/intel/compute-runtime) and IntelPython3.

-The old NUFFT_cpu() and NUFFT_legacy() will be kept in the system for compatibility.

**v2019.2.1-2019.2.2**

-Remove lsmr as scipy 1.13 has caused unknown error. 

**v2019.2.0**

- Bump

**v2019.1.2**

- BUGFIX: fix the loss of accuracy in cSpmvh(). Replace the group/local by global memory (the group/local sizes have caused the unknown run-time behaviour on cuda)

**v2019.1.1**

- Refactor the re_subroutine.py

- Adopt tensor form

**v0.4.0.0**

- 0.4.0.0 is a beta version.

- Major updates for the NUFFT_hsa class, including memory reduction and split-radix. Multiple NUFFT_hsa() using cuda backend becomes possible, by pushing the context to the top of the stack when a method is called. 

- Tested in Windows 10 with PyCUDA 2018.1.1, nvidia-driver 417.35, CUDA 9.2, Visual Studio 2015 Community, and Anaconda Python 3.7 64-bit. PyOpenCL in Windows is yet to be tested. 

- Add batch mode.  

  
 
**v0.3.3.12** 

- 0.3.3.12 is a bug-fixed version.

- Removal of the keyword async for compatibility reasons because Reikna has changed the keyword to async_.

**v0.3.3.8**
 
- Bugfix: mm = numpy.tile(mm, [numpy.prod(Jd).astype(int), 1])  to fix the wrong type when numpy.prod(Jd) is not cast as int

- Bugfix: fix rcond=None error in Anaconda 3.6.5 and Numpy 1.13.1 (the recommended None in Numpy 1.14 is backward incompatible with 1.13)

- Bugfix:  indx1 = indx.copy() is replaced by indx1 = list(indx) for Python2 compatibility

**v0.3.3.7**

- Bugfix in 0.3.3.7 Toeplitz is removed from the NUFFT_cpu and NUFFT_gpu to avoid the MemoryError.

**v0.3.3.6**

- Bugfix: correct the error of import. Now import NUFFT_cpu, NUFFT_hsa at the top level.


**v0.3.3**

- Note: GPU support is superseded by Heterogeneous System Architecture (HSA). 

- A variety of nonlinear solvers are provided, including the conjugate gradient method (cg), L1 total-variation ordinary least square (L1TVOLS), and L1 total-variation least absolute deviation (L1TVLAD).

- The CPU version support other nonlinear solvers, lsqr, gmr, cg, bicgstab, bicg, cgs, gmres, lgmres , apart from cg, L1TVOLS and L1TVLAD.

- Support multi-dimensional transform and reconstruction (experimentally).

**v0.3.2.9**

- Experimental support of NVIDIA's graphic processing unit (GPU). 

- The experimental class gpuNUFFT requires pycuda, scikit-cuda, and python-cuda-cffi. scikit-cuda can be installed from standard command.

**v0.3.2.8**

- Tested under Linux and Windows Anaconda3
  
**v0.3**

- Updated setup.py

- Removal of pyfftw due to segfault under some Linux distributions

  
