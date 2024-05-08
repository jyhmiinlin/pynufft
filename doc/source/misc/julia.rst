PyCall: using PyNUFFT in Julia
==============================
Julia is a new high performance language for numerical analysis. 
Julia scripts are dynamically compiled by LLVM compiler. 

The following steps explain how to import pynufft in Julia:

- Install PyCall   

In Julia environment, install PyCall::

   julia> Pkg.add("PyCall")
   julia> using PyCall

- Import PyNUFFT

Then import pynufft::

   julia>  @pyimport pynufft.tests as tests
   julia> tests.test_installation()

   Does pynufft.py exist?  True
   Does om1D.npz exist? True
   Does om2D.npz exist? True
   Does om3D.npz exist? True
   Does phantom_3D_128_128_128.npz exist? True
   Does phantom_256_256.npz exist? True
   Does example_1D.py exist? True
   Does example_2D.py exist? True
   reikna  has been installed.
   pyopencl  has been installed.
   pycuda  has been installed.

- Import NUFFT_cpu and NUFFT_hsa classes:

Now the NUFFT classes can be easily imported as follows::

   julia> @pyimport pynufft
   julia> N_c = pynufft.NUFFT_cpu
   PyObject <class 'pynufft.NUFFT'>
   julia> N_h = pynufft.NUFFT_hsa
   PyObject <class 'pynufft.NUFFT'>


- Plan NUFFT
 
Please refer to the examples in Tutorial.

- Load Julia images and tests

Install Julia TestImages::

   julia> Pkg.add("TestImages")
   ...
   INFO: Download Completed.
   INFO: Package database updated
   INFO: METADATA is out-of-date — you may not have the latest version of TestImages
   INFO: Use `Pkg.update()` to get the latest versions of your packages
   
   julia> using TestImages
   INFO: Precompiling module TestImages.
   
   julia> Pkg.add("ImageView")
   julia> using ImageView
   
   julia> img = testimage("cameraman")
   julia> imshow(img)
   
   
   
   
   
   
   
   

    
