# import torch
import scipy.sparse
import numpy as np  

def test_tf_eager():
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
    A = pynufft.NUFFT_tf_eager()
    A.plan(om, Nd, Kd, Jd)
    
    y_tf = A.forward(image)
    x_tf = A.adjoint(y_tf)
    # print(y_tf.shape)
    # print(x_tf.shape)
    nfft = pynufft.NUFFT()  # CPU
#     print(nfft.processor)
    
    nfft.plan(om, Nd, Kd, Jd)
    y = nfft.forward(image)
    x2 = nfft.adjoint(y)
    print('Forward error between tf and numpy',np.linalg.norm(y_tf - y)/np.linalg.norm(y))
    print('Adjoint Error between tf and numpy', np.linalg.norm(x2 - x_tf)/np.linalg.norm(x2))
        
# test_torch()
# test_tensorflow()
# test_torch_class()
if __name__ == '__main__':
    test_tf_eager()
# test_random_sp()