import numpy
import pynufft

NufftObj = pynufft.NUFFT()

om = numpy.random.randn(1512,1) 
om = numpy.random.randn(1512,1) 
# om is an M x 1 ndarray: locations of M points. *om* is normalized between [-pi, pi]
# Here M = 1512

Nd = (256,)
Kd = (512,)
Jd = (6,)

NufftObj.plan(om, Nd, Kd, Jd) 

# Now test 1D case

import matplotlib.pyplot as pyplot
# Now build a box function
time_data = numpy.zeros(256, )
time_data[96:128+32] = 1.0
# Now display the function
pyplot.plot(time_data)
pyplot.ylim(-1,2)
pyplot.show()


# Forward NUFFT
y = NufftObj.forward(time_data)

# Display the nonuniform spectrum 
pyplot.plot(om,y.real,'.', label='real') 
pyplot.plot(om,y.imag,'r.', label='imag') 
pyplot.legend()
pyplot.show()



# Adjoint NUFFT
x2 = NufftObj.adjoint(y) 
pyplot.plot(x2.real,'.-', label='real') 
pyplot.plot(x2.imag,'r.-', label='imag') 
pyplot.plot(time_data,'k',label='original signal')
pyplot.ylim(-1,2)
pyplot.legend()
pyplot.show()  


# Test inverse method using density compensation 
x3 = NufftObj.solve(y,'dc',maxiter=1) 
pyplot.plot(x3.real,'.-', label='real') 
pyplot.plot(x3.imag,'r.-', label='imag') 
pyplot.plot(time_data,'k',label='original signal')
pyplot.ylim(-1,2)
pyplot.legend()
pyplot.show()
