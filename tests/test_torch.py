# import torch
import scipy.sparse
import numpy as np  
#
# # 或者: from torch import sparse [as 別名]
# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor.
#
#        https://github.com/DSE-MSU/DeepRobust
#     """
#     sparse_mx = sparse_mx.tocoo().astype(np.complex64)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape) 
#
#
# # def sparse_mx_to_tf_sparse_tensor(sparse_mx):
# #     """Convert a scipy sparse matrix to a torch sparse tensor.
# #
# #        https://github.com/DSE-MSU/DeepRobust
# #     """
# #     sparse_mx = sparse_mx.tocoo().astype(np.complex64)
# #     indices = tf.convert_to_tensor(
# #         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
# #     values = tf.convert_to_tensor(sparse_mx.data)
# #     shape = (sparse_mx.shape)
# #     return tf.sparse.SparseTensor(indices, values, shape) 
#
# def sparse_mx_to_tf_sparse_tensor(X):
#     coo = X.tocoo()
#     indices = np.mat([coo.row, coo.col]).transpose()
#     return tf.SparseTensor(indices, coo.data.astype('complex64'), coo.shape)
# def test_random_sp():
#     A = scipy.sparse.rand(20,20,0.1) + 1.0j*scipy.sparse.rand(20,20,0.1)
#     B = sparse_mx_to_torch_sparse_tensor(A)
#
#
#     #print(B.dtype, B.data, A.data)
#     v = np.random.randn(20,)
#     y1 = A.dot(v)
#     y2 = B.mv(torch.tensor(v.astype(np.complex64)))
#
#     print(y1 - np.array(y2))

# from pynufft import NUFFT
# import pynufft
# import numpy
# dtype = numpy.complex64
#
# def test_torch():
#
# #     cm = matplotlib.cm.gray
#     # load example image
#     import pkg_resources
#
#     DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
# #     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
#     import numpy
#
# #     import matplotlib.pyplot
#
#     import scipy
#     import scipy.misc
#
#     image = scipy.misc.ascent()[::2,::2]
#     image=image.astype(numpy.float)/numpy.max(image[...])
#
#     Nd = (256, 256)  # image space size
#     Kd = (512, 512)  # k-space size
#     Jd = (6,6)  # interpolation size
#
#     # load k-space points
#     om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
#
#     nfft = NUFFT()  # CPU
# #     print(nfft.processor)
#
#     nfft.plan(om, Nd, Kd, Jd)
#     y = nfft.forward(image)
#     k = nfft.xx2k(nfft.x2xx(image)).flatten()
#     x2 = nfft.adjoint(y)
#     x3 = nfft.selfadjoint(image)
#     torch_sn = torch.tensor(nfft.sn.astype(np.complex64))
#     x_tmp = torch_sn*torch.tensor(image.astype(np.complex64))
#     k = torch.zeros((512,512),dtype=torch.complex64)
#     k[list(slice(Nd[jj]) for jj in range(0, 2))] = x_tmp
#     k = torch.fft.fftn(k)
#     torch_sp = sparse_mx_to_torch_sparse_tensor(nfft.sp)
#     torch_spH = sparse_mx_to_torch_sparse_tensor(nfft.spH)
#     torch_y = torch_sp.mv(torch.flatten(k))
#     k2 = torch.reshape(torch_spH.mv(torch_y), (512,512))
#     xx = torch.fft.ifftn(k2)[list(slice(Nd[jj]) for jj in range(0, 2))] 
#     # import matplotlib.pyplot
#     # matplotlib.pyplot.imshow(x2.real)
#     # matplotlib.pyplot.show()
#
#
#     x_out = torch_sn*xx
#     # matplotlib.pyplot.imshow(x_out.real)
#     # matplotlib.pyplot.show()
#
#
#     print(np.linalg.norm(torch_y - y)/np.linalg.norm(y))
#     print(np.linalg.norm(x2 - x3)/np.linalg.norm(x2))
#     # print(y)
#
# import tensorflow as tf
# tf.executing_eagerly()
#
# def test_tensorflow():
#
# #     cm = matplotlib.cm.gray
#     # load example image
#     import pkg_resources
#
#     DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
# #     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
#     import numpy
#
# #     import matplotlib.pyplot
#
#     import scipy
#     import scipy.misc
#
#     image = scipy.misc.ascent()[::2,::2]
#     image=image.astype(numpy.float)/numpy.max(image[...])
#
#     Nd = (256, 256)  # image space size
#     Kd = (512, 512)  # k-space size
#     Jd = (6,6)  # interpolation size
#
#     # load k-space points
#     om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
#
#     nfft = NUFFT()  # CPU
# #     print(nfft.processor)
#
#     nfft.plan(om, Nd, Kd, Jd)
#     y = nfft.forward(image)
#     k = nfft.xx2k(nfft.x2xx(image)).flatten()
#     x2 = nfft.adjoint(y)
#     x3 = nfft.selfadjoint(image)
#     tf_sn = tf.convert_to_tensor(nfft.sn, dtype=tf.complex64)
#     x_tmp = tf_sn*image
#     xx = np.zeros(Kd,dtype=np.complex64)
#     xx[list(slice(Nd[jj]) for jj in range(0, 2))] = x_tmp
#     k = tf.signal.fft2d(xx)
#     tf_sp = sparse_mx_to_tf_sparse_tensor(nfft.sp)
#     tf_spH = sparse_mx_to_tf_sparse_tensor(nfft.spH)
#     k = tf.reshape(k, (512*512, 1))#(np.prod(Kd), ))
#     print(k.dtype, tf_sp.dtype)
#     tf_y = tf.sparse.sparse_dense_matmul(tf_sp, k)
#     print(tf_y.shape)
#     k2 = tf.reshape(tf.sparse.sparse_dense_matmul(tf_spH, tf_y), Kd)
#     xx = tf.signal.ifft2d(k2)[list(slice(Nd[jj]) for jj in range(0, 2))] 
#     # import matplotlib.pyplot
#     # matplotlib.pyplot.imshow(x2.real)
#     # matplotlib.pyplot.show()
#
#
#     x_out = tf_sn*xx
#     # matplotlib.pyplot.imshow(x_out.real)
#     # matplotlib.pyplot.show()
#
#
#     print(np.linalg.norm(tf_y[:,0] - y)/np.linalg.norm(y))
#     print(np.linalg.norm(x_out - x3)/np.linalg.norm(x3))
    # print(y)    
def test_torch():
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
    A = pynufft.NUFFT_torch()
    A.plan(om, Nd, Kd, Jd)
    
    y_torch = A.forward(image)
    x_torch = A.adjoint(y_torch)
    
    nfft = pynufft.NUFFT()  # CPU
#     print(nfft.processor)
    
    nfft.plan(om, Nd, Kd, Jd)
    y = nfft.forward(image)
    x2 = nfft.adjoint(y)
    print('Forward Error between torch and numpy', np.linalg.norm(y_torch - y)/np.linalg.norm(y))
    print('Adjoint Error between torch and numpy', np.linalg.norm(x2 - numpy.array(x_torch))/np.linalg.norm(x2))
  
# test_torch()
# test_tensorflow()
if __name__ == '__main__':
    test_torch()
# test_tf_class()
# test_random_sp()