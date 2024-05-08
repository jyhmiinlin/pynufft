Installation under Macbook Apple Silicon
========================

**Macbook Air M1 (MacOS Ventura 13.5.2)**

Apple provides a integrated GPU on its own apple silicon M1 processor. 
The M1 GPU supports OpenCL 1.2.

To identify the existing OpenCL driver, install clinfo from homebrew:

- Install homebrew from terminal ::

   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

- Install clinfo from terminal ::

   brew install clinfo
   
- Check the opencl driver ::

	Number of platforms                               1
	  Platform Name                                   Apple
	  Platform Vendor                                 Apple
	  Platform Version                                OpenCL 1.2 (Jun 23 2023 20:24:12)
	  Platform Profile                                FULL_PROFILE
	  Platform Extensions                             cl_APPLE_SetMemObjectDestructor cl_APPLE_ContextLoggingFunctions cl_APPLE_clut cl_APPLE_query_kernel_names cl_APPLE_gl_sharing cl_khr_gl_event
	
	  Platform Name                                   Apple
	Number of devices                                 1
	  Device Name                                     Apple M1
	  Device Vendor                                   Apple
	  Device Vendor ID                                0x1027f00
	  Device Version                                  OpenCL 1.2 
	  Driver Version                                  1.2 1.0
	  Device OpenCL C Version                         OpenCL C 1.2 
	  Device Type                                     GPU
	  Device Profile                                  FULL_PROFILE
	  Device Available                                Yes
	  Compiler Available                              Yes
	  Linker Available                                Yes
	  Max compute units                               8
	  Max clock frequency                             1000MHz
	  Device Partition                                (core)
	    Max number of sub-devices                     0
	    Supported partition types                     None
	    Supported affinity domains                    (n/a)
	  Max work item dimensions                        3
	  Max work item sizes                             256x256x256
	  Max work group size                             256
	  Preferred work group size multiple (kernel)     32
	  Preferred / native vector sizes                 
	    char                                                 1 / 1       
	    short                                                1 / 1       
	    int                                                  1 / 1       
	    long                                                 1 / 1       
	    half                                                 0 / 0        (n/a)
	    float                                                1 / 1       
	    double                                               1 / 1        (n/a)
	  Half-precision Floating-point support           (n/a)
	  Single-precision Floating-point support         (core)
	    Denormals                                     No
	    Infinity and NANs                             Yes
	    Round to nearest                              Yes
	    Round to zero                                 Yes
	    Round to infinity                             Yes
	    IEEE754-2008 fused multiply-add               Yes
	    Support is emulated in software               No
	    Correctly-rounded divide and sqrt operations  Yes
	  Double-precision Floating-point support         (n/a)
	  Address bits                                    64, Little-Endian
	  Global memory size                              11453251584 (10.67GiB)
	  Error Correction support                        No
	  Max memory allocation                           2147483648 (2GiB)
	  Unified memory for Host and Device              Yes
	  Minimum alignment for any data type             1 bytes
	  Alignment of base address                       32768 bits (4096 bytes)
	  Global Memory cache type                        None
	  Image support                                   Yes
	    Max number of samplers per kernel             32
	    Max size for 1D images from buffer            268435456 pixels
	    Max 1D or 2D image array size                 2048 images
	    Base address alignment for 2D image buffers   256 bytes
	    Pitch alignment for 2D image buffers          256 pixels
	    Max 2D image size                             16384x16384 pixels
	    Max 3D image size                             2048x2048x2048 pixels
	    Max number of read image args                 128
	    Max number of write image args                8
	  Local memory type                               Local
	  Local memory size                               32768 (32KiB)
	  Max number of constant args                     31
	  Max constant buffer size                        1073741824 (1024MiB)
	  Max size of kernel argument                     4096 (4KiB)
	  Queue properties                                
	    Out-of-order execution                        No
	    Profiling                                     Yes
	  Prefer user sync for interop                    Yes
	  Profiling timer resolution                      1000ns
	  Execution capabilities                          
	    Run OpenCL kernels                            Yes
	    Run native kernels                            No
	  printf() buffer size                            1048576 (1024KiB)
	  Built-in kernels                                (n/a)
	  Device Extensions                               cl_APPLE_SetMemObjectDestructor cl_APPLE_ContextLoggingFunctions cl_APPLE_clut cl_APPLE_query_kernel_names cl_APPLE_gl_sharing cl_khr_gl_event cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_3d_image_writes cl_khr_image2d_from_buffer cl_khr_depth_images 
	
	NULL platform behavior
	  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  Apple
	  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   Success [P0]
	  clCreateContext(NULL, ...) [default]            Success [P0]
	  clCreateContextFromType(NULL, CL_DEVICE_TYPE_DEFAULT)  Success (1)
	    Platform Name                                 Apple
	    Device Name                                   Apple M1
	  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No devices found in platform
	  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  Success (1)
	    Platform Name                                 Apple
	    Device Name                                   Apple M1
	  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No devices found in platform
	  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  Invalid device type for platform
	  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  Success (1)
	    Platform Name                                 Apple
	    Device Name                                   Apple M1
   
   
- Run pynufft tests ::

	Python 3.11.5 (main, Aug 24 2023, 15:09:45) [Clang 14.0.3 (clang-1403.0.22.14.1)] on darwin
	Type "help", "copyright", "credits" or "license" for more information.
	>>> from pynufft import tests as t
	>>> t.test_init()
	No cuda device found. Check your pycuda installation.
	device name =  <pyopencl.Device 'Apple M1' on 'Apple' at 0x1027f00>
	0.024686098098754883
	0.0167756986618042
	error gx2= 2.1132891e-07
	error gy= 1.1100628500806993e-07
	acceleration= 1.4715391946662406
	7.563086748123169 4.348098039627075
	acceleration in solver= 1.7394011540668561




