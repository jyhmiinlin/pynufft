import numpy
from ..src._helper import helper#, helper1

def _init__cpu(self):
        """
        Constructor.
 
        :param None:
        :type None: Python NoneType
        :return: NUFFT: the pynufft_hsa.NUFFT instance
        :rtype: NUFFT: the pynufft_hsa.NUFFT class
        :Example:
 
        >>> from pynufft import NUFFT
        >>> NufftObj = NUFFT()
        """
        self.dtype = numpy.complex64  # : initial value: numpy.complex64
        self.debug = 0  #: initial value: 0
        self.Nd = ()  # : initial value: ()
        self.Kd = ()  # : initial value: ()
        self.Jd = ()  #: initial value: ()
        self.ndims = 0  # : initial value: 0
        self.ft_axes = ()  # : initial value: ()
        self.batch = None  # : initial value: None
        pass    

def _plan_cpu(self, om, Nd, Kd, Jd, ft_axes=None):
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

    self.ndims = len(Nd)  # : initial value: len(Nd)
    if ft_axes is None:
        ft_axes = range(0, self.ndims)
    self.ft_axes = ft_axes  # default: all axes (range(0, self.ndims)

    self.st = helper.plan(om, Nd, Kd, Jd, ft_axes=ft_axes,
                          format='CSR')

    self.Nd = self.st['Nd']  # backup
    self.Kd = self.st['Kd']
    # backup
    self.sn = numpy.asarray(self.st['sn'].astype(self.dtype), order='C')

#     if batch is None:  # single-coil
    self.parallel_flag = 0
    self.batch = 1

#     else:  # multi-coil
#         self.parallel_flag = 1
#         self.batch = batch

#     if self.parallel_flag is 1:
#         self.multi_Nd = self.Nd + (self.batch, )
#         self.uni_Nd = self.Nd + (1, )
#         self.multi_Kd = self.Kd + (self.batch, )
#         self.multi_M = (self.st['M'], ) + (self.batch, )
#         self.multi_prodKd = (numpy.prod(self.Kd), self.batch)
#         self.sn = numpy.reshape(self.sn, self.Nd + (1,))

#     elif self.parallel_flag is 0:
    self.Nd = self.Nd  # + (self.Reps, )
    self.Nd = self.Nd
    self.Kd = self.Kd  # + (self.Reps, )
    self.M = (self.st['M'], )
    self.prodKd = (numpy.prod(self.Kd), )

    # Calculate the density compensation function
    self.sp = self.st['p'].copy().tocsr()
    
    self.spH = (self.st['p'].getH().copy()).tocsr()
    
    self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
    self.Jdprod = numpy.int32(numpy.prod(self.st['Jd']))
    del self.st['p'], self.st['sn']

    self.NdCPUorder, self.KdCPUorder, self.nelem = helper.preindex_copy(
        self.st['Nd'],
        self.st['Kd'])
#     self.volume = {}
#     self.volume['cpu_coil_profile'] = numpy.ones(self.multi_Nd)

    return 0


def _precompute_sp_cpu(self):
    """

    Private: Precompute adjoint (gridding) and Toepitz interpolation
             matrix.

    :param None:
    :type None: Python Nonetype
    :return: self: instance
    """
    try:
        W0 = numpy.ones((self.st['M'],), dtype=numpy.complex64)
        W = self.xx2k(self.adjoint(W0))
        self.W = (W*W.conj())**0.5
        del W0
        del W
    except:
        print("errors occur in self.precompute_sp()")
        raise




def _solve_cpu(self, y, solver=None, *args, **kwargs):
    """
    Solve NUFFT.
    :param y: data, numpy.complex64. The shape = (M,) or (M, batch)
    :param solver: 'cg', 'L1TVOLS', 'lsmr', 'lsqr', 'dc', 'bicg',
                   'bicgstab', 'cg', 'gmres','lgmres'
    :param maxiter: the number of iterations
    :type y: numpy array, dtype = numpy.complex64
    :type solver: string
    :type maxiter: int
    :return: numpy array with size.
            The shape = Nd ('L1TVOLS') or  Nd 
            ('lsmr', 'lsqr', 'dc','bicg','bicgstab','cg', 'gmres','lgmres')
    """
    from ..linalg.solve_cpu import solve
    x2 = solve(self,  y,  solver, *args, **kwargs)
    return x2  # solve(self,  y,  solver, *args, **kwargs)

def _forward_cpu(self, x):
    """
    Forward NUFFT on CPU

    :param x: The input numpy array, with the size of Nd
    :type: numpy array with the dtype of numpy.complex64
    :return: y: The output numpy array, with the size of (M,)
    :rtype: numpy array with the dtype of numpy.complex64
    """
    y = self._k2y_cpu(self._xx2k_cpu(self._x2xx_cpu(x)))

    return y

def _adjoint_cpu(self, y):
    """
    Adjoint NUFFT on CPU

    :param y: The input numpy array, with the size of (M,) 
    :type: numpy array with the dtype of numpy.complex64
    :return: x: The output numpy array,
                with the size of Nd or Nd 
    :rtype: numpy array with the dtype of numpy.complex64
    """
    x = self._xx2x_cpu(self._k2xx_cpu(self._y2k_cpu(y)))

    return x

def _selfadjoint_one2many2one_cpu_deprecated(self, x):
    y2 = self._forward_one2many_cpu(x)
    x2 = self._adjoint_many2one_cpu(y2)
    del y2
    return x2

def _selfadjoint_cpu(self, x):
    """
    selfadjoint NUFFT on CPU

    :param x: The input numpy array, with size=Nd
    :type: numpy array with dtype =numpy.complex64
    :return: x: The output numpy array, with size=Nd
    :rtype: numpy array with dtype =numpy.complex64
    """
    # x2 = self.adjoint(self.forward(x))

    x2 = self._xx2x_cpu(self._k2xx_cpu(self._k2y2k_cpu(self._xx2k_cpu(self._x2xx_cpu(x)))))

    return x2

def _selfadjoint2_cpu(self, x):
    try:
        x2 = self._k2xx_cpu(self.W * self._xx2k_cpu(x))
    except:
        self._precompute_sp_cpu()
        x2 = self._k2xx_cpu(self.W * self._xx2k_cpu(x))
    return x2

def _x2xx_cpu(self, x):
    """
    Private: Scaling on CPU
    Inplace multiplication of self.x_Nd by the scaling factor self.sn.
    """
    xx = x * self.sn
    return xx

def _xx2k_cpu(self, xx):
    """
    Private: oversampled FFT on CPU

    Firstly, zeroing the self.k_Kd array
    Second, copy self.x_Nd array to self.k_Kd array by cSelect
    Third, inplace FFT
    """
    
    output_x = numpy.zeros(self.Kd, dtype=self.dtype, order='C')

    for bat in range(0, self.batch):
        output_x.ravel()[self.KdCPUorder * self.batch + bat] = xx.ravel()[
            self.NdCPUorder * self.batch + bat]

    k = numpy.fft.fftn(output_x, axes=self.ft_axes)
    
    return k

def _xx2k_one2one_cpu(self, xx):
    """
    Private: oversampled FFT on CPU

    First, zeroing the self.k_Kd array
    Second, copy self.x_Nd array to self.k_Kd array by cSelect
    Third, inplace FFT
    """
    
    output_x = numpy.zeros(self.st['Kd'], dtype=self.dtype, order='C')

    # for bat in range(0, self.batch):
    output_x.ravel()[self.KdCPUorder] = xx.ravel()[self.NdCPUorder]

    k = numpy.fft.fftn(output_x, axes=self.ft_axes)
    
    return k

def _k2vec_cpu(self, k):
    k_vec = numpy.reshape(k, self.prodKd, order='C')
    return k_vec

def _vec2y_cpu(self, k_vec):
    '''
    gridding:
    '''
    y = self.sp.dot(k_vec)
    # y = self.st['ell'].spmv(k_vec)

    return y

def _k2y_cpu(self, k):
    """
    Private: interpolation by the Sparse Matrix-Vector Multiplication
    """
    y = self._vec2y_cpu(self._k2vec_cpu(k))
    # numpy.reshape(self.st['p'].dot(Xk), (self.st['M'], ), order='F')
    return y

def _y2vec_cpu(self, y):
    '''
   regridding non-uniform data (unsorted vector)
    '''
    # k_vec = self.st['p'].getH().dot(y)
    k_vec = self.spH.dot(y)
    # k_vec = self.st['ell'].spmvH(y)

    return k_vec

def _vec2k_cpu(self, k_vec):
    '''
    Sorting the vector to k-spectrum Kd array
    '''
    k = numpy.reshape(k_vec, self.Kd, order='C')

    return k

def _y2k_cpu(self, y):
    """
    Private: gridding by the Sparse Matrix-Vector Multiplication
    """
    k_vec = self._y2vec_cpu(y)
    k = self._vec2k_cpu(k_vec)
    return k

def _k2xx_cpu(self, k):
    """
    Private: the inverse FFT and image cropping (which is the reverse of
             _xx2k() method)
    """
#         dd = numpy.size(self.Kd)

    k = numpy.fft.ifftn(k, axes=self.ft_axes)
    xx = numpy.zeros(self.Nd, dtype=self.dtype, order='C')
    for bat in range(0, self.batch):
        xx.ravel()[self.NdCPUorder * self.batch + bat] = k.ravel()[
            self.KdCPUorder * self.batch + bat]
#         xx = xx[crop_slice_ind(self.Nd)]
    return xx


def _k2xx_one2one_cpu(self, k):
    """
    Private: the inverse FFT and image cropping
             (which is the reverse of _xx2k() method)
    """
#         dd = numpy.size(self.Kd)

    k = numpy.fft.ifftn(k, axes=self.ft_axes)
    xx = numpy.zeros(self.st['Nd'], dtype=self.dtype, order='C')
    # for bat in range(0, self.batch):
    xx.ravel()[self.NdCPUorder] = k.ravel()[self.KdCPUorder]
    # xx = xx[crop_slice_ind(self.Nd)]
    return xx

def _xx2x_cpu(self, xx):
    """
    Private: rescaling, which is identical to the  _x2xx() method
    """
    x = self._x2xx_cpu(xx)
    return x

def _k2y2k_cpu(self, k):
    """
    Private: the integrated interpolation-gridding by the Sparse
             Matrix-Vector Multiplication
    """

    Xk = self._k2vec_cpu(k)
    k = self._y2vec_cpu(self._vec2y_cpu(Xk))
    k = self._vec2k_cpu(k)
    return k


## host code
def _forward_host(self, x):
    gx = self.to_device(x)
    y = self._forward_device(gx).get()
    return y

def _forward_legacy_host(self, x):
    gx = self.to_device(x)
    y = self._forward_legacy(gx).get()
    return y

def _adjoint_host(self, y):
    gy = self.to_device(y)
    gx = self._adjoint_device(gy)
    return gx.get()

def _adjoint_legacy_host(self, y):
    gy = self.to_device(y)
    gx = self._adjoint_legacy(gy)
    return gx.get()
    
def _selfadjoint_host(self, x):
    gx = self.to_device(x)
    gx2 = self._selfadjoint_device(gx)
    return gx2.get()
    
def _selfadjoint_legacy_host(self, x):
    gx = self.to_device(x)
    gx2 = self._selfadjoint_legacy(gx)
    return gx2.get()
    
def _solve_host(self, y, *args, **kwargs):
    gy = self.to_device(y)
    x = self._solve_device(gy, *args, **kwargs)
    return x.get()

def _solve_legacy_host(self, y, *args, **kwargs):
    gy = self.to_device(y)
    x = self._solve_legacy(gy, *args, **kwargs)
    return x.get()

def _xx2k_host(self, xx):
    gxx = self.to_device(xx)
    gk = self._xx2k_device(gxx)
    return gk.get()


def _k2xx_host(self, k):
    gk = self.to_device(k)
    gxx = self._k2xx_device(gk)
    return gxx.get()

def _x2xx_host(self, x):
    gx = self.to_device(x)
    gxx = self._x2xx_device(gx)
    return gxx.get()

def _xx2x_host(self, x):
    gx = self.to_device(x)
    gxx = self._xx2x_device(gx)
    return gxx.get()

def _k2y_host(self, k):
    gk = self.to_device(k)
    gy = self._k2y_device(gk)
    return gy.get()

def _y2k_host(self, y):
    gy = self.to_device(y)
    gk = self._y2k_device(gy)
    return gk.get()

def _k2y_legacy_host(self, k):
    gk = self.to_device(k)
    gy = self._k2y_legacy(gk)
    return gy.get()

def _y2k_legacy_host(self, y):
    gy = self.to_device(y)
    gk = self._y2k_legacy(gy)
    return gk.get()
