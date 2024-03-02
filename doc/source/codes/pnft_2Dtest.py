import numpy
# from pynufft import NUFFT
import pynufft
NufftObj = pynufft.NUFFT()

# load the data folder
import pkg_resources

# find the relative path of data folder
DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')

# now load the om locations
om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']

# om is an Mx2 numpy.ndarray
print(om)

# display om
import matplotlib.pyplot as pyplot
pyplot.plot(om[:,0], om[:,1])
pyplot.title('2D trajectory of M points')
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.show()


Nd = (256,256) # image dimension
Kd = (512,512) # k-spectrum dimension
Jd = (6,6) # interpolator size

NufftObj.plan(om, Nd, Kd, Jd) 


# load image from scipy.misc.face()
import scipy.misc
import matplotlib.cm as cm
image = scipy.misc.ascent()[::2,::2]
image=image.astype(numpy.float)/numpy.max(image[...])
pyplot.imshow(image, cmap=cm.gray)
pyplot.show()


# Forward NUFFT transform
y = NufftObj.forward(image)

# Adjoint NUFFT
k =   NufftObj.y2k(y)
import matplotlib.colors
k_show = numpy.fft.fftshift(k)
pyplot.imshow(numpy.abs(k_show), cmap=cm.gray, norm=matplotlib.colors.Normalize(0, 1e+3))
pyplot.show()

# Inverse transform using density compensation inverse_DC()
x3 = NufftObj.solve(y,'dc',maxiter=1)
x3_display = x3*1.0/numpy.max(x3[...].real)
pyplot.imshow(x3_display.real,cmap=cm.gray)
pyplot.show()
