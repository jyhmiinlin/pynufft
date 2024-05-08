"""
An example of multiprocessing with NUFFT using PyCUDA backend, 
wrapped inside the atomic_NUFFT wrapper class.  
The two processes are running on two CPU cores.
Each process creates one NUFFT and offloads the computations to GPU.
nvidia-smi confirms that two python programs are using the GPU.
"""

import numpy
from pynufft import NUFFT, helper
import scipy.misc
import matplotlib.pyplot
import multiprocessing
import os 
    


class atomic_NUFFT:
    def __init__(self, om, Nd, Kd, Jd, device_indx):
        """
        This caches the parameters only.
        Any other GPU related stuffs are carried out in run()
        """
        self.om = om
        self.Nd = Nd
        self.Kd = Kd
        self.Jd = Jd
#         self.API = API
        self.device_indx = device_indx
        
    def run(self, x, cpu_cores):
        """
        In this method, the NUFFT are created and executed on a fixed CPU core.
        """
        pid= os.getpid()
        print('pid=', pid)
        os.system("taskset -p -c %d-%d %d" % (cpu_cores[0], cpu_cores[1], pid))
        """
        Control the CPU affinity. Otherwise the process on one core can be switched to another core.
        """

        # create NUFFT
#         NUFFT = NUFFT(self.API, )
        
        # plan the NUFFT
        
        device_list = helper.device_list()  
        self.NUFFT = NUFFT(device_list[self.device_indx])
        self.NUFFT.plan(self.om, self.Nd, self.Kd, self.Jd)
        # send the image to device
        gx = self.NUFFT.to_device(x)
        
        # carry out 10000 forward transform
        for pp in range(0, 10000):
            gy = self.NUFFT._forward_device(gx)

        # return the object
        return gy.get()

Nd = (256,256)
Kd = (512,512)
Jd = (6,6)
om = numpy.random.randn(35636, 2) 
x = scipy.misc.ascent()[::2,::2]
om1 = om[om[:,0]>0, :]
om2 = om[om[:,0]<=0, :]

# create pool
pool = multiprocessing.Pool(2)

# create the list to receive the return values
results = []

# Now enter the first process
# This is the standard multiprocessing Pool
D = atomic_NUFFT(om1, Nd, Kd, Jd, 0)
# async won't obstruct the next line of code
result = pool.apply_async(D.run, args = (x, (0,3)))
# the result is appended
results.append(result)

# Now enter the second process
# This is the standard multiprocessing Pool
D = atomic_NUFFT(om2, Nd, Kd, Jd, 0)
# Non-obstructive
result = pool.apply_async(D.run, args = (x, (4,7)))
results.append(result)

# closing the pool 
pool.close()
pool.join()

# results are appended
# Now print the outputs
result1 = results[0].get()
result2 = results[1].get()

# check CPU results

NUFFT_cpu1 = NUFFT()
NUFFT_cpu1.plan(om1, Nd, Kd, Jd)
y1 = NUFFT_cpu1.forward(x)
print('norm = ', numpy.linalg.norm(y1 - result1) / numpy.linalg.norm(y1))

NUFFT_cpu2 = NUFFT()
NUFFT_cpu2.plan(om2, Nd, Kd, Jd)
y2 = NUFFT_cpu2.forward(x)
print('norm = ', numpy.linalg.norm(y2 - result2) / numpy.linalg.norm(y2))

