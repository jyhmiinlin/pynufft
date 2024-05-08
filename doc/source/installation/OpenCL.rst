Installation of OpenCL
======================

OpenCL is one of the backends that PyNUFFT supports. Up to the present, PyNUFFT has used OpenCL-1.2. One missing feature of OpenCL-1.2 is `atomicAdd` for the array with floating point numbers.   PyNUFFT makes use of `atomic_cmpxchg` (compare and exchange) to implement the atomic_add_float subroutine, which can be seen in the `pynufft.src.re_subroutine.atomic_add`. This code has appeared in many resources, e.g. `http://simpleopencl.blogspot.com/2013/05/atomic-operations-and-floats-in-opencl.html` and `https://github.com/clMathLibraries/clSPARSE/blob/master/src/library/kernels/csrmv_adaptive.cl`.

Note that the OpenCL standard is still evolving and all of the OpenCL supports may change quickly. Old sdk may not work with the newest Intel chipsets. Please try different versions of the hardware and software.

The current compiler version is gcc version 7.3.0. Other compilers may be used on the target system but I haven't tested any of them. 

**Intel OpenCL**

Intel HD graphics after the Skylake generation usually support OpenCL as long as the suitable intel-sdk is installed.  

One OpenCL example is a Gigabyte aero 15 Gentoo Linux on a machine with Intel Corporation HD Graphics 530 (rev 06). The dev-util/intel-ocl-sdk-4.4.0.117-r1 installed the `intel_sdk_for_ocl_applications_2014_ubuntu_4.4.0.117_x64.tgz` opencl package. The `clinfo` command shows ::

     Platform Name                                   Intel(R) OpenCL    
     Number of devices                                 1
     Device Name                                     Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz
     Device Vendor                                   Intel(R) Corporation
     Device Vendor ID                                0x8086
     Device Version                                  OpenCL 1.2 (Build 117)
     Driver Version                                  1.2.0.117
     Device OpenCL C Version                         OpenCL C 1.2 
     Device Type                                     CPU
     Device Profile                                  FULL_PROFILE
     Device Available                                Yes
     Compiler Available                              Yes
     Linker Available                                Yes
     Max compute units                               8
     Max clock frequency                             2600MHz
     Device Partition                                (core)
       Max number of sub-devices                     8
       Supported partition types                     by counts, equally, by names (Intel)
       Supported affinity domains                    (n/a)
     Max work item dimensions                        3
     Max work item sizes                             8192x8192x8192
     Max work group size                             8192
     Preferred work group size multiple              128
     Preferred / native vector sizes                 
       char                                                 1 / 32      
       short                                                1 / 16      
       int                                                  1 / 8       
       long                                                 1 / 4       
       half                                                 0 / 0        (n/a)
       float                                                1 / 8       
       double                                               1 / 4        (cl_khr_fp64)
     Half-precision Floating-point support           (n/a)
     Single-precision Floating-point support         (core)
       Denormals                                     Yes
       Infinity and NANs                             Yes
       Round to nearest                              Yes
       Round to zero                                 No
       Round to infinity                             No
       IEEE754-2008 fused multiply-add               No
       Support is emulated in software               No
       Correctly-rounded divide and sqrt operations  No
     Double-precision Floating-point support         (cl_khr_fp64)
       Denormals                                     Yes
       Infinity and NANs                             Yes
       Round to nearest                              Yes
       Round to zero                                 Yes
       Round to infinity                             Yes
       IEEE754-2008 fused multiply-add               Yes
       Support is emulated in software               No
     Address bits                                    64, Little-Endian
     Global memory size                              33613447168 (31.3GiB)
     Error Correction support                        No
     Max memory allocation                           8403361792 (7.826GiB)
     Unified memory for Host and Device              Yes
     Minimum alignment for any data type             128 bytes
     Alignment of base address                       1024 bits (128 bytes)
     Global Memory cache type                        Read/Write
     Global Memory cache size                        262144 (256KiB)
     Global Memory cache line size                   64 bytes
     Image support                                   Yes
       Max number of samplers per kernel             480
       Max size for 1D images from buffer            525210112 pixels
       Max 1D or 2D image array size                 2048 images
       Max 2D image size                             16384x16384 pixels
       Max 3D image size                             2048x2048x2048 pixels
       Max number of read image args                 480
       Max number of write image args                480
     Local memory type                               Global
     Local memory size                               32768 (32KiB)
     Max number of constant args                     480
     Max constant buffer size                        131072 (128KiB)
     Max size of kernel argument                     3840 (3.75KiB)
     Queue properties                                
       Out-of-order execution                        Yes
       Profiling                                     Yes
       Local thread execution (Intel)                Yes
     Prefer user sync for interop                    No
     Profiling timer resolution                      1ns
     Execution capabilities                          
       Run OpenCL kernels                            Yes
       Run native kernels                            Yes
       SPIR versions                                 1.2
     printf() buffer size                            1048576 (1024KiB)
     Built-in kernels                                (n/a)
     Device Extensions                               cl_khr_icd cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_byte_addressable_store cl_khr_spir cl_intel_exec_by_local_thread cl_khr_depth_images cl_khr_3d_image_writes cl_khr_fp64 
       

Pure CPU system without Intel HD graphics may require the newest Intel SDK for OpenCL `https://software.intel.com/en-us/intel-opencl` and `https://software.intel.com/en-us/articles/opencl-drivers`. One pure CPU system with Intel i7 7900X can make use of Intel Studio 2019. 

**Nvidia OpenCL**

NVIDIA also supports OpenCL 1.2. A successful installation made use of nvidia-driver 417.18 and CUDA-SDK-9.2.88 and gcc 7.3.0. clinfo shows ::

     Platform Name                                   NVIDIA CUDA
   Number of devices                                 1
     Device Name                                     GeForce GTX 1060
     Device Vendor                                   NVIDIA Corporation
     Device Vendor ID                                0x10de
     Device Version                                  OpenCL 1.2 CUDA
     Driver Version                                  415.18
     Device OpenCL C Version                         OpenCL C 1.2 
     Device Type                                     GPU
     Device Topology (NV)                            PCI-E, 01:00.0
     Device Profile                                  FULL_PROFILE
     Device Available                                Yes
     Compiler Available                              Yes
     Linker Available                                Yes
     Max compute units                               10
     Max clock frequency                             1670MHz
     Compute Capability (NV)                         6.1
     Device Partition                                (core)
       Max number of sub-devices                     1
       Supported partition types                     None
       Supported affinity domains                    (n/a)
     Max work item dimensions                        3
     Max work item sizes                             1024x1024x64
     Max work group size                             1024
     Preferred work group size multiple              32
     Warp size (NV)                                  32
     Preferred / native vector sizes                 
       char                                                 1 / 1       
       short                                                1 / 1       
       int                                                  1 / 1       
       long                                                 1 / 1       
       half                                                 0 / 0        (n/a)
       float                                                1 / 1       
       double                                               1 / 1        (cl_khr_fp64)
     Half-precision Floating-point support           (n/a)
     Single-precision Floating-point support         (core)
       Denormals                                     Yes
       Infinity and NANs                             Yes
       Round to nearest                              Yes
       Round to zero                                 Yes
       Round to infinity                             Yes
       IEEE754-2008 fused multiply-add               Yes
       Support is emulated in software               No
       Correctly-rounded divide and sqrt operations  Yes
     Double-precision Floating-point support         (cl_khr_fp64)
       Denormals                                     Yes
       Infinity and NANs                             Yes
       Round to nearest                              Yes
       Round to zero                                 Yes
       Round to infinity                             Yes
       IEEE754-2008 fused multiply-add               Yes
       Support is emulated in software               No
     Address bits                                    64, Little-Endian
     Global memory size                              6373572608 (5.936GiB)
     Error Correction support                        No
     Max memory allocation                           1593393152 (1.484GiB)
     Unified memory for Host and Device              No
     Integrated memory (NV)                          No
     Minimum alignment for any data type             128 bytes
     Alignment of base address                       4096 bits (512 bytes)
     Global Memory cache type                        Read/Write
     Global Memory cache size                        163840 (160KiB)
     Global Memory cache line size                   128 bytes
     Image support                                   Yes
       Max number of samplers per kernel             32
       Max size for 1D images from buffer            134217728 pixels
       Max 1D or 2D image array size                 2048 images
       Max 2D image size                             16384x32768 pixels
       Max 3D image size                             16384x16384x16384 pixels
       Max number of read image args                 256
       Max number of write image args                16
     Local memory type                               Local
     Local memory size                               49152 (48KiB)
     Registers per block (NV)                        65536
     Max number of constant args                     9
     Max constant buffer size                        65536 (64KiB)
     Max size of kernel argument                     4352 (4.25KiB)
     Queue properties                                
       Out-of-order execution                        Yes
       Profiling                                     Yes
     Prefer user sync for interop                    No
     Profiling timer resolution                      1000ns
     Execution capabilities                          
       Run OpenCL kernels                            Yes
       Run native kernels                            No
       Kernel execution timeout (NV)                 No
     Concurrent copy and kernel execution (NV)       Yes
       Number of async copy engines                  2
     printf() buffer size                            1048576 (1024KiB)
     Built-in kernels                                (n/a)
     Device Extensions                               cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer
 
**AMD GPU**

AMD has a very good support for OpenCL. See AMDGPU-PRO driver. AMD usually performs very well for fp64 (double precision PyNUFFT is available on request).

**Open-source Intel Compute OpenCL (Intel-NEO and Intel SDK)**

Try to install Intel's proprietary OpenCL sdk. 
 










