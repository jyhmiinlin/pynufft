# import torch
import scipy.sparse
import numpy as np  

def test_cupy():
    import pynufft
    import pkg_resources
    
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
    import numpy
    
#     import matplotlib.pyplot
    
    import scipy
    import scipy.misc

    image = scipy.misc.ascent()[::2,::2]
    image=image.astype(float)/numpy.max(image[...])

    Nd = (256, 256)  # image space size
    Kd = (512, 512)  # k-space size
    Jd = (6,6)  # interpolation size

    # load k-space points
    om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
    A = pynufft.NUFFT_cupy()
    A.plan(om, Nd, Kd, Jd)
    import cupy
    y_cupy = A.forward(cupy.asarray(image))
    x_cupy = A.adjoint(y_cupy)
    
    nfft = pynufft.NUFFT()  # CPU
#     print(nfft.processor)
    
    nfft.plan(om, Nd, Kd, Jd)
    y = nfft.forward(image)
    x2 = nfft.adjoint(y)
    print('Forward Error between cupy and numpy', np.linalg.norm(y_cupy.get() - y)/np.linalg.norm(y))
    print('Adjoint Error between cupy and numpy', np.linalg.norm(x2 - numpy.array(x_cupy.get()))/np.linalg.norm(x2))
  
# test_torch()
# test_tensorflow()
if __name__ == '__main__':
    test_cupy()
# test_tf_class()
# test_random_sp()