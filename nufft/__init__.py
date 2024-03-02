"""
NUFFT class
=======================================
"""
from __future__ import absolute_import
import numpy
import warnings
import scipy.sparse
import numpy.fft
#import scipy.signal
import scipy.linalg
import scipy.special
from functools import wraps as _wraps

from importlib import import_module
# from ..linalg.nufft_cpu import NUFFT_cpu
# from ..linalg.nufft_hsa import NUFFT_hsa

# import torch
# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor.
#
#        https://github.com/DSE-MSU/DeepRobust
#     """
#     sparse_mx = sparse_mx.tocoo().astype(numpy.complex64)
#     indices = torch.from_numpy(
#         numpy.vstack((sparse_mx.row, sparse_mx.col)).astype(numpy.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape) 


def push_cuda_context(hsa_method):
    """
    Decorator: Push cude context to the top of the stack for current use
    Add @push_cuda_context before the methods of NUFFT_device()
    """
    @_wraps(hsa_method)
    def wrapper(*args, **kwargs):
        try:
            args[0].thr._context.push()
        except:
            pass
        return hsa_method(*args, **kwargs)
    return wrapper

class NUFFT_cupy:
    """
    A tentative torch interface
    """
    
    
    def __init__(self):
        self.dtype = numpy.complex64  # : initial value: numpy.complex64
        self.debug = 0  #: initial value: 0
        self.Nd = ()  # : initial value: ()
        self.Kd = ()  # : initial value: ()
        self.Jd = ()  #: initial value: ()
        self.ndims = 0  # : initial value: 0
        self.ft_axes = ()  # : initial value: ()
        self.batch = None  # : initial value: None
        # self.processor='torch'
        self.cupy = import_module('cupy')
        

    def plan(self, om, Nd, Kd, Jd):
        """
        Plan the NUFFT object with the geometry provided.
    
        :param om: The M off-grid locates in the frequency domain,
                    which is normalized between [-pi, pi]
        :param Nd: The matrix size of the equispaced image.
                   Example: Nd=(256,256) for a 2D image;
                             Nd = (128,128,128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid.
                   Example: Kd=(512,512) for 2D image;
                            Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size.
                   Example: Jd=(6,6) for 2D image;
                            Jd = (6,6,6) for a 3D image
        :param ft_axes: (Optional) The axes for Fourier transform.
                        The default is all axes if 'None' is given.
        :param batch: (Optional) Batch mode.
                     If the batch is provided, the last appended axis is the number
                     of identical NUFFT to be transformed.
                     The default is 'None'.
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: None, or tuple with optional integer elements.
        :returns: 0
        :rtype: int, float
    
        :ivar Nd: initial value: Nd
        :ivar Kd: initial value: Kd
        :ivar Jd: initial value: Jd
        :ivar ft_axes: initial value: None
    
        :Example:
    
        >>> from pynufft import NUFFT
        >>> NufftObj = NUFFT()
        >>> NufftObj.plan(om, Nd, Kd, Jd)
    
        or
    
        >>> NufftObj.plan(om, Nd, Kd, Jd, ft_axes)
    
        """
        from ..src._helper import helper#, helper1
        self.ndims = len(Nd)  # : initial value: len(Nd)
        ft_axes = tuple(jj for jj in range(0, self.ndims))
        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes=ft_axes,
                              format='CSR')
    
        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        # backup
        # self.sn = numpy.asarray(self.st['sn'].astype(self.dtype), order='C')
    
    
        self.sn = self.cupy.asarray(numpy.asarray(self.st['sn'].astype(self.dtype), order='C'))
    
        
        self.cupy_sp = self.cupy._cupyx.scipy.sparse.csr_matrix(self.st['p'])
        # torch.sparse.FloatTensor(i, v, torch.Size(shape))#.to_dense()
        
        # i2 = torch.LongTensor(numpy.vstack((coo.col, coo.row)))
        self.cupy_spH = self.cupy._cupyx.scipy.sparse.csr_matrix(self.st['p'].getH().copy())
        # self.spH = torch.sparse.FloatTensor(i2, v, torch.Size((shape[1], shape[0])))
        
        
        # self.spH = (self.st['p'].getH().copy()).tocsr()
        
        # self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        # self.Jdprod = numpy.int32(numpy.prod(self.st['Jd']))
        # del self.st['p'], self.st['sn']
        #
        # self.NdCPUorder, self.KdCPUorder, self.nelem = helper.preindex_copy(
        #     self.st['Nd'],
        #     self.st['Kd'])
    #     self.volume = {}
    #     self.volume['cpu_coil_profile'] = numpy.ones(self.multi_Nd)
    
        return 0
    def forward(self, x):
        xx = self.x2xx(x)
        k = self.xx2k(xx)
        y = self.k2y(k)
        # y = self.k2y(self.xx2k(self.x2xx))
        return y
    def adjoint(self, y):
        k = self.y2k(y)
        xx = self.k2xx(k)
        x = self.xx2x(xx)
        # x = self.xx2x(self.k2xx(self.y2k(y)))
        return x
    def x2xx(self, x):
        xx = self.sn*x
        # xx = torch.mul(x, self.sn)
        return xx
    
    def xx2k(self,xx):
        k = self.cupy.zeros(self.Kd, dtype=numpy.complex64)
        k[tuple(slice(self.Nd[jj]) for jj in range(0, self.ndims))] = xx
        k = self.cupy.fft.fftn(k)
        return k
    
    def k2y(self, k):
        torch_y = self.cupy_sp.dot(k.flatten())
        return torch_y
    
    def y2k(self, cupy_y):
        k = self.cupy.reshape(self.cupy_spH.dot(cupy_y), self.Kd)
        return k
    
    def k2xx(self, k):
        xx = self.cupy.fft.ifftn(k)[tuple(slice(self.Nd[jj]) for jj in range(0, self.ndims))] 
        return xx
    
    def xx2x(self,xx):
        return self.x2xx(xx)
    


class NUFFT_torch:
    """
    A tentative torch interface
    """
    
    
    def __init__(self):
        self.dtype = numpy.complex64  # : initial value: numpy.complex64
        self.debug = 0  #: initial value: 0
        self.Nd = ()  # : initial value: ()
        self.Kd = ()  # : initial value: ()
        self.Jd = ()  #: initial value: ()
        self.ndims = 0  # : initial value: 0
        self.ft_axes = ()  # : initial value: ()
        self.batch = None  # : initial value: None
        # self.processor='torch'
        self.torch = import_module('torch')
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor.
        
           https://github.com/DSE-MSU/DeepRobust
        """
        sparse_mx = sparse_mx.tocoo().astype(numpy.complex64)
        indices = self.torch.from_numpy(
            numpy.vstack((sparse_mx.row, sparse_mx.col)).astype(numpy.int64))
        values = self.torch.from_numpy(sparse_mx.data)
        shape = self.torch.Size(sparse_mx.shape)
        return self.torch.sparse.FloatTensor(indices, values, shape) 

    def plan(self, om, Nd, Kd, Jd):
        """
        Plan the NUFFT object with the geometry provided.
    
        :param om: The M off-grid locates in the frequency domain,
                    which is normalized between [-pi, pi]
        :param Nd: The matrix size of the equispaced image.
                   Example: Nd=(256,256) for a 2D image;
                             Nd = (128,128,128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid.
                   Example: Kd=(512,512) for 2D image;
                            Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size.
                   Example: Jd=(6,6) for 2D image;
                            Jd = (6,6,6) for a 3D image
        :param ft_axes: (Optional) The axes for Fourier transform.
                        The default is all axes if 'None' is given.
        :param batch: (Optional) Batch mode.
                     If the batch is provided, the last appended axis is the number
                     of identical NUFFT to be transformed.
                     The default is 'None'.
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: None, or tuple with optional integer elements.
        :returns: 0
        :rtype: int, float
    
        :ivar Nd: initial value: Nd
        :ivar Kd: initial value: Kd
        :ivar Jd: initial value: Jd
        :ivar ft_axes: initial value: None
    
        :Example:
    
        >>> from pynufft import NUFFT
        >>> NufftObj = NUFFT()
        >>> NufftObj.plan(om, Nd, Kd, Jd)
    
        or
    
        >>> NufftObj.plan(om, Nd, Kd, Jd, ft_axes)
    
        """
        from ..src._helper import helper#, helper1
        self.ndims = len(Nd)  # : initial value: len(Nd)
        ft_axes = tuple(jj for jj in range(0, self.ndims))
        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes=ft_axes,
                              format='CSR')
    
        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        # backup
        # self.sn = numpy.asarray(self.st['sn'].astype(self.dtype), order='C')
    
    
        self.sn = self.torch.from_numpy(numpy.asarray(self.st['sn'].astype(self.dtype), order='C'))
    
        
        self.torch_sp = self.sparse_mx_to_torch_sparse_tensor(self.st['p'])
        # torch.sparse.FloatTensor(i, v, torch.Size(shape))#.to_dense()
        
        # i2 = torch.LongTensor(numpy.vstack((coo.col, coo.row)))
        self.torch_spH = self.sparse_mx_to_torch_sparse_tensor(self.st['p'].getH().copy())
        # self.spH = torch.sparse.FloatTensor(i2, v, torch.Size((shape[1], shape[0])))
        
        
        # self.spH = (self.st['p'].getH().copy()).tocsr()
        
        # self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        # self.Jdprod = numpy.int32(numpy.prod(self.st['Jd']))
        # del self.st['p'], self.st['sn']
        #
        # self.NdCPUorder, self.KdCPUorder, self.nelem = helper.preindex_copy(
        #     self.st['Nd'],
        #     self.st['Kd'])
    #     self.volume = {}
    #     self.volume['cpu_coil_profile'] = numpy.ones(self.multi_Nd)
    
        return 0
    def forward(self, x):
        xx = self.x2xx(x)
        k = self.xx2k(xx)
        y = self.k2y(k)
        # y = self.k2y(self.xx2k(self.x2xx))
        return y
    def adjoint(self, y):
        k = self.y2k(y)
        xx = self.k2xx(k)
        x = self.xx2x(xx)
        # x = self.xx2x(self.k2xx(self.y2k(y)))
        return x
    def x2xx(self, x):
        xx = self.sn*x
        # xx = torch.mul(x, self.sn)
        return xx
    
    def xx2k(self,xx):
        k = self.torch.zeros(self.Kd, dtype=self.torch.complex64)
        k[list(slice(self.Nd[jj]) for jj in range(0, self.ndims))] = xx
        k = self.torch.fft.fftn(k)
        return k
    
    def k2y(self, k):
        torch_y = self.torch_sp.mv(self.torch.flatten(k))
        return torch_y
    
    def y2k(self, torch_y):
        k = self.torch.reshape(self.torch_spH.mv(torch_y), self.Kd)
        return k
    
    def k2xx(self, k):
        xx = self.torch.fft.ifftn(k)[list(slice(self.Nd[jj]) for jj in range(0, self.ndims))] 
        return xx
    
    def xx2x(self,xx):
        return self.x2xx(xx)
    

class NUFFT_tf_eager:
    """
    A tentative torch interface
    """
    
    
    def __init__(self):
        self.dtype = numpy.complex64  # : initial value: numpy.complex64
        self.debug = 0  #: initial value: 0
        self.Nd = ()  # : initial value: ()
        self.Kd = ()  # : initial value: ()
        self.Jd = ()  #: initial value: ()
        self.ndims = 0  # : initial value: 0
        self.ft_axes = ()  # : initial value: ()
        self.batch = None  # : initial value: None
        # self.processor='torch'
        self.tf = import_module('tensorflow')
        self.tf.executing_eagerly()
        
    def sparse_mx_to_tf_sparse_tensor(self, X):
        coo = X.tocoo()
        indices = numpy.mat([coo.row, coo.col]).transpose()
        return self.tf.SparseTensor(indices, coo.data.astype('complex64'), coo.shape)

    def plan(self, om, Nd, Kd, Jd):
        """
        Plan the NUFFT object with the geometry provided.
    
        :param om: The M off-grid locates in the frequency domain,
                    which is normalized between [-pi, pi]
        :param Nd: The matrix size of the equispaced image.
                   Example: Nd=(256,256) for a 2D image;
                             Nd = (128,128,128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid.
                   Example: Kd=(512,512) for 2D image;
                            Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size.
                   Example: Jd=(6,6) for 2D image;
                            Jd = (6,6,6) for a 3D image
        :param ft_axes: (Optional) The axes for Fourier transform.
                        The default is all axes if 'None' is given.
        :param batch: (Optional) Batch mode.
                     If the batch is provided, the last appended axis is the number
                     of identical NUFFT to be transformed.
                     The default is 'None'.
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: None, or tuple with optional integer elements.
        :returns: 0
        :rtype: int, float
    
        :ivar Nd: initial value: Nd
        :ivar Kd: initial value: Kd
        :ivar Jd: initial value: Jd
        :ivar ft_axes: initial value: None
    
        :Example:
    
        >>> from pynufft import NUFFT
        >>> NufftObj = NUFFT()
        >>> NufftObj.plan(om, Nd, Kd, Jd)
    
        or
    
        >>> NufftObj.plan(om, Nd, Kd, Jd, ft_axes)
    
        """
        from ..src._helper import helper#, helper1
        self.ndims = len(Nd)  # : initial value: len(Nd)
        ft_axes = tuple(jj for jj in range(0, self.ndims))
        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes=ft_axes,
                              format='CSR')
    
        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        # backup
        # self.sn = numpy.asarray(self.st['sn'].astype(self.dtype), order='C')
    
    
        # self.sn = self.torch.from_numpy(numpy.asarray(self.st['sn'].astype(self.dtype), order='C'))
        self.sn = self.tf.convert_to_tensor(self.st['sn'], dtype=self.tf.complex64)
        
        self.tf_sp = self.sparse_mx_to_tf_sparse_tensor(self.st['p'])
        # torch.sparse.FloatTensor(i, v, torch.Size(shape))#.to_dense()
        
        # i2 = torch.LongTensor(numpy.vstack((coo.col, coo.row)))
        self.tf_spH = self.sparse_mx_to_tf_sparse_tensor(self.st['p'].getH().copy())
        # self.spH = torch.sparse.FloatTensor(i2, v, torch.Size((shape[1], shape[0])))
        
        
        # self.spH = (self.st['p'].getH().copy()).tocsr()
        
        # self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        # self.Jdprod = numpy.int32(numpy.prod(self.st['Jd']))
        # del self.st['p'], self.st['sn']
        #
        # self.NdCPUorder, self.KdCPUorder, self.nelem = helper.preindex_copy(
        #     self.st['Nd'],
        #     self.st['Kd'])
    #     self.volume = {}
    #     self.volume['cpu_coil_profile'] = numpy.ones(self.multi_Nd)
    
        return 0
    def forward(self, x):
        xx = self.x2xx(x)
        k = self.xx2k(xx)
        y = self.k2y(k)
        # y = self.k2y(self.xx2k(self.x2xx))
        return y
    def adjoint(self, y):
        k = self.y2k(y)
        xx = self.k2xx(k)
        x = self.xx2x(xx)
        # x = self.xx2x(self.k2xx(self.y2k(y)))
        return x
    def x2xx(self, x):
        xx = self.sn*self.tf.convert_to_tensor(x, dtype=self.tf.complex64)
        # xx = torch.mul(x, self.sn)
        return xx
    
    def xx2k(self,xx):
        x_tmp = numpy.zeros(self.Kd, dtype=numpy.complex64)
        x_tmp[tuple(slice(self.Nd[jj]) for jj in range(0, self.ndims))] = xx
        k = self.tf.convert_to_tensor(x_tmp, dtype=self.tf.complex64)
        if self.ndims == 1:
            k = self.tf.signal.fft(k)
        if self.ndims == 2:
            k = self.tf.signal.fft2d(k)
        elif self.ndims == 3:
            k = self.tf.signal.fft3d(k)
        return k
    
    def k2y(self, k):
        k = self.tf.reshape(k, (numpy.prod(self.Kd), 1))
        # torch_y = self.torch_sp.mv(self.torch.flatten(k))
        tf_y = self.tf.sparse.sparse_dense_matmul(self.tf_sp, k)
        return tf_y[:,0]
    
    def y2k(self, tf_y):
        tf_y = self.tf.reshape(tf_y, tf_y.shape + (1,))
        k2 = self.tf.reshape(self.tf.sparse.sparse_dense_matmul(self.tf_spH, tf_y), self.Kd)
        # k = self.torch.reshape(self.torch_spH.mv(torch_y), self.Kd)
        return k2
    
    def k2xx(self, k):
        if self.ndims == 1:
            k = self.tf.signal.ifft(k)
        if self.ndims == 2:
            k = self.tf.signal.ifft2d(k)
        elif self.ndims == 3:
            k = self.tf.signal.ifft3d(k)
        xx = k[list(slice(self.Nd[jj]) for jj in range(0, self.ndims))]
        # xx = self.torch.fft.ifftn(k)[list(slice(self.Nd[jj]) for jj in range(0, self.ndims))] 
        return xx
    
    def xx2x(self,xx):
        return self.x2xx(xx)

class NUFFT:
    """
    NUFFT class
    =======================================
    A super class of cpu and gpu NUFFT functions. 
    
    Note: NUFFT does not inherit NUFFT_cpu (deprecated) and NUFFT_hsa (deprecated).
    """
    #import cpu codes
    from ._nufft_class_methods_cpu import _init__cpu, _plan_cpu, _precompute_sp_cpu, _solve_cpu, _forward_cpu, _adjoint_cpu, _selfadjoint_cpu, _selfadjoint2_cpu, _x2xx_cpu, _xx2k_cpu, _xx2k_one2one_cpu, _k2vec_cpu, _vec2y_cpu, _k2y_cpu, _y2vec_cpu, _vec2k_cpu, _y2k_cpu, _k2xx_cpu, _k2xx_one2one_cpu, _xx2x_cpu, _k2y2k_cpu
    # import host codes
    from ._nufft_class_methods_cpu import  _forward_host,  _adjoint_host, _selfadjoint_host,  _solve_host, _xx2k_host, _k2xx_host, _x2xx_host, _xx2x_host, _k2y_host,  _y2k_host
    # import device codes
    from ._nufft_class_methods_device import _init__device, _plan_device,  _set_wavefront_device, _offload_device, to_device, to_host, _x2xx_device, _xx2k_device, _k2y_device, _y2k_device, _k2xx_device, _xx2x_device,  _selfadjoint_device, _forward_device,  _adjoint_device, release, _solve_device, release
    
    # legacy codes (csr format for device)
    from ._nufft_class_methods_device import _y2k_legacy, _k2y_legacy, _forward_legacy, _adjoint_legacy, _selfadjoint_legacy, _plan_legacy, _offload_legacy, _solve_legacy
    from ._nufft_class_methods_cpu import _k2y_legacy_host,  _y2k_legacy_host, _selfadjoint_legacy_host, _forward_legacy_host, _adjoint_legacy_host,  _solve_legacy_host
    
    def __init__(self, device_indx=None, legacy=None):
        """
        Constructor.

        :param None:
        :type None: Python NoneType
        :return: NUFFT: the pynufft.NUFFT instance
        :rtype: NUFFT: the pynufft.NUFFT class
        :Example:

        >>> from pynufft import NUFFT
        >>> NufftObj = NUFFT()
        
        or 
        
        >>> from pynufft import NUFFT, helper
        >>> device = helper.device_list()[0]
        >>> NufftObj = NUFFT(device) # for first acceleration device in the system
        
        """
        if device_indx is None:
            self._init__cpu()
            self.processor = 'cpu'
        else:
            if legacy is True:
                self._init__device(device_indx)
                self.processor = 'hsa_legacy'
            else:
                self._init__device(device_indx)
                self.processor = 'hsa'
            
    def __del__(self):
        if self.processor is 'hsa' or 'hsa_legacy':
            self.release()
        else:
            pass
                
    def plan(self,  *args, **kwargs):
        """
        Plan the NUFFT object with the geometry provided.

        :param om: The M off-grid locates in the frequency domain,
                    which is normalized between [-pi, pi]
        :param Nd: The matrix size of the equispaced image.
                   Example: Nd=(256,256) for a 2D image;
                             Nd = (128,128,128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid.
                   Example: Kd=(512,512) for 2D image;
                            Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size.
                   Example: Jd=(6,6) for 2D image;
                            Jd = (6,6,6) for a 3D image
        :param ft_axes: (Optional) The axes for Fourier transform.
                        The default is all axes if 'None' is given.
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: None, or tuple with optional integer elements.
        :returns: 0
        :rtype: int, float

        :ivar Nd: initial value: Nd
        :ivar Kd: initial value: Kd
        :ivar Jd: initial value: Jd
        :ivar ft_axes: initial value: None

        :Example:

        >>> from pynufft import NUFFT
        >>> NufftObj = NUFFT()
        >>> NufftObj.plan(om, Nd, Kd, Jd)

        or

        >>> NufftObj.plan(om, Nd, Kd, Jd, ft_axes)

        """        
        func = {'cpu': self._plan_cpu,
                    'hsa': self._plan_device,
                    'hsa_legacy': self._plan_legacy}
        return func.get(self.processor)(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """
        Forward NUFFT (host code)

        :param x: The input numpy array, with the size of Nd 
        :type: numpy array with the dtype of numpy.complex64
        :return: y: The output numpy array, with the size of (M,) 
        :rtype: numpy array with the dtype of numpy.complex64
        """        
        func = {'cpu': self._forward_cpu, 
                    'hsa': self._forward_host,
                    'hsa_legacy': self._forward_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def adjoint(self, *args, **kwargs):
        """
        Adjoint NUFFT (host code)

        :param y: The input numpy array, with the size of (M,) 
        :type: numpy array with the dtype of numpy.complex64
        :return: x: The output numpy array,
                    with the size of Nd or Nd 
        :rtype: numpy array with the dtype of numpy.complex64
        """        
        func = {'cpu': self._adjoint_cpu,
                'hsa': self._adjoint_host,
                'hsa_legacy': self._adjoint_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def selfadjoint(self, *args, **kwargs):
        """
        selfadjoint NUFFT (host code)

        :param x: The input numpy array, with size=Nd
        :type: numpy array with dtype =numpy.complex64
        :return: x: The output numpy array, with size=Nd
        :rtype: numpy array with dtype =numpy.complex64
        """
        func = {'cpu': self._selfadjoint_cpu,
                'hsa': self._selfadjoint_host,
                'hsa_legacy': self._selfadjoint_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def solve(self, *args, **kwargs):
        """
        Solve NUFFT (host code)
        :param y: data, numpy.complex64. The shape = (M,) 
        :param solver: 'cg', 'L1TVOLS', 'lsmr', 'lsqr', 'dc', 'bicg',
                       'bicgstab', 'cg', 'gmres','lgmres'
        :param maxiter: the number of iterations
        :type y: numpy array, dtype = numpy.complex64
        :type solver: string
        :type maxiter: int
        :return: numpy array with size Nd.
        """        
        func = {'cpu': self._solve_cpu,
                'hsa': self._solve_host,
                'hsa_legacy': self._solve_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def xx2k(self, *args, **kwargs):
        func = {'cpu': self._xx2k_cpu,
                'hsa': self._xx2k_host,
                'hsa_legacy': self._xx2k_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def k2xx(self, *args, **kwargs):
        func = {'cpu': self._k2xx_cpu,
                'hsa': self._k2xx_host,
                'hsa_legacy': self._k2xx_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def x2xx(self, *args, **kwargs):
        func = {'cpu': self._x2xx_cpu,
                'hsa': self._x2xx_host,
                'hsa_legacy': self._x2xx_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def xx2x(self, *args, **kwargs):
        func = {'cpu': self._xx2x_cpu,
                'hsa': self._xx2x_host,
                'hsa_legacy': self._xx2x_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def k2y(self, *args, **kwargs):
        func = {'cpu': self._k2y_cpu,
                'hsa': self._k2y_host,
                'hsa_legacy': self._k2y_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def y2k(self, *args, **kwargs):
        func = {'cpu': self._y2k_cpu,
                'hsa': self._y2k_host,
                'hsa_legacy': self._y2k_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def k2yk2(self, *args, **kwargs):
        func = {'cpu': self._k2yk2_cpu,
                'hsa': self._k2yk2_host,
                'hsa_legacy': self._k2yk2_host}
        return func.get(self.processor)(*args, **kwargs)
    
#     def adjoint_many2one(self, *args, **kwargs):
#         func = {'cpu': self._adjoint_many2one_cpu,
#                 'hsa': self._adjoint_many2one_host,
#                 'hsa_legacy': self._adjoint_many2one_legacy_host}
#         return func.get(self.processor)(*args, **kwargs)
#     
#     def forward_one2many(self, *args, **kwargs):
#         func = {'cpu': self._forward_one2many_cpu,
#                 'hsa': self._forward_one2many_host,
#                 'hsa_legacy': self._forward_one2many_legacy_host}
#         return func.get(self.processor)(*args, **kwargs)
    
#     def selfadjoint_one2many2one(self, *args, **kwargs):
#         func = {'cpu': self._selfadjoint_one2many2one_cpu,
#                 'hsa': self._selfadjoint_one2many2one_host,
#                 'hsa_legacy': self._selfadjoint_one2many2one_legacy_host}
#         return func.get(self.processor)(*args, **kwargs)   
    
    def k2xx_one2one(self, *args, **kwargs):
        func = {'cpu':self._k2xx_one2one_cpu}
        return func.get(self.processor)(*args, **kwargs)
    
    def xx2k_one2one(self, *args, **kwargs):
        func = {'cpu':self._xx2k_one2one_cpu}
        return func.get(self.processor)(*args, **kwargs) 
    
    def k2y2k(self, *args, **kwargs):
        func = {'cpu': self._k2y2k_cpu}
        return func.get(self.processor)(*args, **kwargs)
#     def set_sense(self, *args, **kwargs):
#         func = {'cpu': self._set_sense_cpu,
#                 'hsa': self._set_sense_host,
#                 'hsa_legacy': self._set_sense_host}
#         return func.get(self.processor)(*args, **kwargs)
#     def reset_sense(self, *args, **kwargs):
#         func = {'cpu': self._reset_sense_cpu,
#                 'hsa': self._reset_sense_host,
#                 'hsa_legacy': self._reset_sense_host}
#         return func.get(self.processor)(*args, **kwargs)
    
