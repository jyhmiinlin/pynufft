Installation under Windows 10
=============================

PyNUFFT has been tested under the Windows 10 home edition, with Anaconda3-2021.11 64-bit, PyCUDA 2021.1 
from official pip, Microsoft Visual Studio 2022 Community, and CUDA 11.6.  

**Pre-requisites**

- A NVIDIA GPU and a clean Windows 10

- Install the Microsoft Visual Studio 2022 Community (check cl.exe, see troubleshooting).

- Install CUDA-11.6 and the related driver.

Now open command prompt `cmd`, type ::

   nvcc -V

You will see ::

   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2018 NVIDIA Corporation
   Built on Tue_Jun_12_23:09_12_Central_Daylight_time_2018
   Cuda compilation tool, release 9.2, V9.2.148    
   



- Troubleshotting: Add the environmental variable path.

If the system cannot find `cl.exe` when you type `cl`: ::

   C:\Users\User>cl
   `cl` is not recognized as an internal or external command, 
   operable program or batch file.
   
this error is due to the fact that Visual Studio has not been added to the system path, so the system cannot find cl. 
 
Add Path "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\MSVC\14.31.31103\\bin\\Hostx64\x64" to system variables. (see troubleshotting)
 
Once Visual Studio has been added to the system, open Windows cmd and it should find cl.exe ::
   
   C:\Users\User>cl
   Microsoft (R) C/C++ Optimizing Compiler Version 19.0024215.1 for x86
   Copyright (C) Microsoft Corpooration. All rights reserved.
   
   usage: cl [ option... ] filename... [ /link linkoption... ]

without the earlier error message. If the error persists, the path must be modified again. 

**Installation of Anaconda3**

-  Now install Anaconda3. I downloaded Anaconda3-2021.11 64-bit. Once this is done you can follow the general installation procedure as described above. 

**Installation of PyCUDA, Reikna and PyNUFFT**

- Open Anaconda3 command prompt, type::

   pip install pycuda
   pip install reikna
   pip install pynufft
   
- Test PyNUFFT::

   python
   from pynufft import tests
   tests.test_init()
   





