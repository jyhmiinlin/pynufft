'''
=================
3D wireframe plot
=================

A very basic demonstration of a wireframe plot.
'''

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
#X, Y, Z = axes3d.get_test_data(0.05)

#print(Y)
import numpy

from pynufft import NUFFT_cpu

NUFFT = NUFFT_cpu()
om = numpy.random.randn(1,2)

Nd = (128,128)
Kd = (512,512)
Jd = (12,12)
NUFFT.plan(om, Nd, Kd, Jd)
data = NUFFT.sp.data
print(data)
Z = data.reshape(Jd)
# Plot a basic wireframe.
#ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
X, Y = numpy.meshgrid(numpy.arange(0, 12), numpy.arange(0,12))
#ax.plot_wireframe(X, Y, Z)

#plt.show()

NUFFT = NUFFT_cpu()
om = numpy.random.randn(1,2)

Nd = (128, )
Kd = (512, )
Jd = (12, )
NUFFT.plan(om, Nd, Kd, Jd)
data = NUFFT.sp.data

plt.plot(numpy.arange(0,12), data.real)

plt.show()
