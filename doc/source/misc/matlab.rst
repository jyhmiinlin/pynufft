Install MATLAB NUFFT from Michigan Image Reconstruction Toolbox (MIRT), by Professor Jeff Fessler and his students
==================================================================================================================

MATLAB NUFFT can be downloaded from http://web.eecs.umich.edu/~fessler/irt/fessler.tgz .

**Installing NUFFT** 

- Unpack the package. 

- Change directory into "irt"::
    
    cd irt

- Enter octave program::

    octave
    
- If you have matlab, type matlab instead of octave::

    matlab

- Setup the environment in octave or matlab using the setup script::

    setup
  
- The following message shows that everything is ready:: 

   The variable "irtdir" is not set, so trying default, assuming
   that you launched matlab from the irt install directory.
   You may need to edit setup.m or adjust your path otherwise.
   ...
   Path setup for irt appears to have succeeded.
   
- Run all of the tests::

   cd example
   test_all_example
   
- Everything is fine if all tests passed::

   all 24 tests passed!
   
- Using st=nufft_init(om, Nd, Jd, Kd, nshifts) to build a structure. Then use y=nufft(image, st) to transform the image to k-space. 

   
****     