"""
NUFFT CPU class (deprecated)
=======================================
"""

from __future__ import absolute_import
import numpy
import scipy.sparse
import numpy.fft
import scipy.linalg
import scipy.special

from ..src._helper import helper, helper1


class NUFFT_cpu:
    """
    Class NUFFT_cpu
   """
    def __init__(self):
        """
        Constructor.

        :param None:
        :type None: Python NoneType
        :return: NUFFT: the pynufft_hsa.NUFFT instance
        :rtype: NUFFT: the pynufft_hsa.NUFFT class
        :Example:

        >>> from pynufft import NUFFT_cpu
        >>> NufftObj = NUFFT_cpu()
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

    def plan(self, om, Nd, Kd, Jd, ft_axes=None):
        """
        Plan the NUFFT_cpu object with the geometry provided.

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
        :type batch: None, or integer
        :returns: 0
        :rtype: int, float

        :ivar Nd: initial value: Nd
        :ivar Kd: initial value: Kd
        :ivar Jd: initial value: Jd
        :ivar ft_axes: initial value: None
        :ivar batch: initial value: None

        :Example:

        >>> from pynufft import NUFFT_cpu
        >>> NufftObj = NUFFT_cpu()
        >>> NufftObj.plan(om, Nd, Kd, Jd)

        or

        >>> NufftObj.plan(om, Nd, Kd, Jd, ft_axes, batch)

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

        if batch is None:  # single-coil
            self.parallel_flag = 0
            self.batch = 1

        else:  # multi-coil
            self.parallel_flag = 1
            self.batch = batch

        if self.parallel_flag == 1:
            self.multi_Nd = self.Nd + (self.batch, )
            self.uni_Nd = self.Nd + (1, )
            self.multi_Kd = self.Kd + (self.batch, )
            self.multi_M = (self.st['M'], ) + (self.batch, )
            self.multi_prodKd = (numpy.prod(self.Kd), self.batch)
            self.sn = numpy.reshape(self.sn, self.Nd + (1,))

        elif self.parallel_flag == 0:
            self.multi_Nd = self.Nd  # + (self.Reps, )
            self.uni_Nd = self.Nd
            self.multi_Kd = self.Kd  # + (self.Reps, )
            self.multi_M = (self.st['M'], )
            self.multi_prodKd = (numpy.prod(self.Kd), )

        # Calculate the density compensation function
        self.sp = self.st['p'].copy().tocsr()
        
        self.spH = (self.st['p'].getH().copy()).tocsr()
        
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.Jdprod = numpy.int32(numpy.prod(self.st['Jd']))
        del self.st['p'], self.st['sn']

        self.NdCPUorder, self.KdCPUorder, self.nelem = helper.preindex_copy(
            self.st['Nd'],
            self.st['Kd'])
        self.volume = {}
        self.volume['cpu_coil_profile'] = numpy.ones(self.multi_Nd)

        return 0

    def _precompute_sp(self):
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

    def reset_sense(self):
        self.volume['cpu_coil_profile'].fill(1.0)

    def set_sense(self, coil_profile):

        self.volume = {}

        if coil_profile.shape == self.Nd + (self.batch, ):
            self.volume['cpu_coil_profile'] = coil_profile
        else:
            print('The shape of coil_profile might be wrong')
            print('coil_profile.shape = ', coil_profile.shape)
            print('shape of Nd + (batch, ) = ', self.Nd + (self.batch, ))

    def forward_one2many(self, x):
        """
        Assume x.shape = self.Nd

        """

#         try:
        x2 = x.reshape(self.uni_Nd, order='C')*self.volume['cpu_coil_profile']
#         except:
#         x2 = x
        y2 = self.forward(x2)

        return y2

    def adjoint_many2one(self, y):
        """
        Assume y.shape = self.multi_M
        """
        x2 = self.adjoint(y)
        x = x2*self.volume['cpu_coil_profile'].conj()
        try:
            x3 = numpy.mean(x, axis=self.ndims)
        except:
            x3 = x
        del x

        return x3

    def solve(self, y, solver=None, *args, **kwargs):
        """
        Solve NUFFT_cpu.
        :param y: data, numpy.complex64. The shape = (M,) or (M, batch)
        :param solver: 'cg', 'L1TVOLS', 'lsmr', 'lsqr', 'dc', 'bicg',
                       'bicgstab', 'cg', 'gmres','lgmres'
        :param maxiter: the number of iterations
        :type y: numpy array, dtype = numpy.complex64
        :type solver: string
        :type maxiter: int
        :return: numpy array with size.
                The shape = Nd ('L1TVOLS') or  Nd + (batch,)
                ('lsmr', 'lsqr', 'dc','bicg','bicgstab','cg', 'gmres','lgmres')
        """
        from ..linalg.solve_cpu import solve
        x2 = solve(self,  y,  solver, *args, **kwargs)
        return x2  # solve(self,  y,  solver, *args, **kwargs)

    def forward(self, x):
        """
        Forward NUFFT on CPU

        :param x: The input numpy array, with the size of Nd or Nd + (batch,)
        :type: numpy array with the dtype of numpy.complex64
        :return: y: The output numpy array, with the size of (M,) or (M, batch)
        :rtype: numpy array with the dtype of numpy.complex64
        """
        y = self.k2y(self.xx2k(self.x2xx(x)))

        return y

    def adjoint(self, y):
        """
        Adjoint NUFFT on CPU

        :param y: The input numpy array, with the size of (M,) or (M, batch)
        :type: numpy array with the dtype of numpy.complex64
        :return: x: The output numpy array,
                    with the size of Nd or Nd + (batch, )
        :rtype: numpy array with the dtype of numpy.complex64
        """
        x = self.xx2x(self.k2xx(self.y2k(y)))

        return x

    def selfadjoint_one2many2one(self, x):
        y2 = self.forward_one2many(x)
        x2 = self.adjoint_many2one(y2)
        del y2
        return x2

    def selfadjoint(self, x):
        """
        selfadjoint NUFFT on CPU

        :param x: The input numpy array, with size=Nd
        :type: numpy array with dtype =numpy.complex64
        :return: x: The output numpy array, with size=Nd
        :rtype: numpy array with dtype =numpy.complex64
        """
        # x2 = self.adjoint(self.forward(x))

        x2 = self.xx2x(self.k2xx(self.k2y2k(self.xx2k(self.x2xx(x)))))

        return x2

    def selfadjoint2(self, x):
        try:
            x2 = self.k2xx(self.W * self.xx2k(x))
        except:
            self._precompute_sp()
            x2 = self.k2xx(self.W * self.xx2k(x))
        return x2

    def x2xx(self, x):
        """
        Private: Scaling on CPU
        Inplace multiplication of self.x_Nd by the scaling factor self.sn.
        """
        xx = x * self.sn
        return xx

    def xx2k(self, xx):
        """
        Private: oversampled FFT on CPU

        Firstly, zeroing the self.k_Kd array
        Second, copy self.x_Nd array to self.k_Kd array by cSelect
        Third, inplace FFT
        """
        
        output_x = numpy.zeros(self.multi_Kd, dtype=self.dtype, order='C')

        for bat in range(0, self.batch):
            output_x.ravel()[self.KdCPUorder * self.batch + bat] = xx.ravel()[
                self.NdCPUorder * self.batch + bat]

        k = numpy.fft.fftn(output_x, axes=self.ft_axes)
        
        return k

    def xx2k_one2one(self, xx):
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

    def k2vec(self, k):
        k_vec = numpy.reshape(k, self.multi_prodKd, order='C')
        return k_vec

    def vec2y(self, k_vec):
        '''
        gridding:
        '''
        y = self.sp.dot(k_vec)
        # y = self.st['ell'].spmv(k_vec)

        return y

    def k2y(self, k):
        """
        Private: interpolation by the Sparse Matrix-Vector Multiplication
        """
        y = self.vec2y(self.k2vec(k))
        # numpy.reshape(self.st['p'].dot(Xk), (self.st['M'], ), order='F')
        return y

    def y2vec(self, y):
        '''
       regridding non-uniform data (unsorted vector)
        '''
        # k_vec = self.st['p'].getH().dot(y)
        k_vec = self.spH.dot(y)
        # k_vec = self.st['ell'].spmvH(y)

        return k_vec

    def vec2k(self, k_vec):
        '''
        Sorting the vector to k-spectrum Kd array
        '''
        k = numpy.reshape(k_vec, self.multi_Kd, order='C')

        return k

    def y2k(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        """
        k_vec = self.y2vec(y)
        k = self.vec2k(k_vec)
        return k

    def k2xx(self, k):
        """
        Private: the inverse FFT and image cropping (which is the reverse of
                 _xx2k() method)
        """
#         dd = numpy.size(self.Kd)

        k = numpy.fft.ifftn(k, axes=self.ft_axes)
        xx = numpy.zeros(self.multi_Nd, dtype=self.dtype, order='C')
        for bat in range(0, self.batch):
            xx.ravel()[self.NdCPUorder * self.batch + bat] = k.ravel()[
                self.KdCPUorder * self.batch + bat]
#         xx = xx[crop_slice_ind(self.Nd)]
        return xx

    def k2xx_one2one(self, k):
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

    def xx2x(self, xx):
        """
        Private: rescaling, which is identical to the  _x2xx() method
        """
        x = self.x2xx(xx)
        return x

    def k2y2k(self, k):
        """
        Private: the integrated interpolation-gridding by the Sparse
                 Matrix-Vector Multiplication
        """

        Xk = self.k2vec(k)
        # k = self.spHsp.dot(Xk)
        # k = self.spH.dot(self.sp.dot(Xk))
        k = self.y2vec(self.vec2y(Xk))
        k = self.vec2k(k)
        return k