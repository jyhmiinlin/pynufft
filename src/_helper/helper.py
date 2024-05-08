"""
Helper functions
=======================================
"""


import numpy
dtype = numpy.complex64
import scipy

def create_laplacian_kernel(nufft):
    """
    Create the multi-dimensional laplacian kernel in k-space

    :param nufft: the NUFFT object
    :return: uker: the multi-dimensional laplacian kernel in k-space (no fft shift used)
    :rtype: numpy ndarray
    """
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#         # related to constraint
#===============================================================================
    uker = numpy.zeros(nufft.st['Kd'][:],dtype=numpy.complex64,order='C')
    n_dims= numpy.size(nufft.st['Nd'])

    ################################
    #    Multi-dimensional laplacian kernel (generalize the above 1D - 3D to multi-dimensional arrays)
    ################################

    indx = [slice(0, 1) for ss in range(0, n_dims)] # create the n_dims dimensional slice which are all zeros
    uker[tuple(indx)] = - 2.0*n_dims # Equivalent to  uker[0,0,0] = -6.0
    for pp in range(0,n_dims):
#         indx1 = indx.copy() # indexing the 1 Only for Python3
        indx1 = list(indx)# indexing; adding list() for Python2/3 compatibility
        indx1[pp] = 1
        uker[tuple(indx1)] = 1
#         indx1 = indx.copy() # indexing the -1  Only for Python3
        indx1 = list(indx)# indexing the 1 Python2/3 compatible
        indx1[pp] = -1
        uker[tuple(indx1)] = 1
    ################################
    #    FFT of the multi-dimensional laplacian kernel
    ################################
    uker =numpy.fft.fftn(uker) #, self.nufftobj.st['Kd'], range(0,numpy.ndim(uker)))
    return uker
def indxmap_diff(Nd):
    """
    Preindixing for rapid image gradient.
    Diff(x) = x.flat[d_indx[0]] - x.flat
    Diff_t(x) =  x.flat[dt_indx[0]] - x.flat

    :param Nd: the dimension of the image
    :type Nd: tuple with integers
    :return: d_indx: image gradient
    :return:  dt_indx:  the transpose of the image gradient
    :rtype: d_indx: lists with numpy ndarray
    :rtype: dt_indx: lists with numpy ndarray
    """

    ndims = len(Nd)
    Ndprod = numpy.prod(Nd)
    mylist = numpy.arange(0, Ndprod).astype(numpy.int32)
    mylist = numpy.reshape(mylist, Nd)
    d_indx = []
    dt_indx = []
    for pp in range(0, ndims):
        d_indx = d_indx + [ numpy.reshape(   numpy.roll(  mylist, +1 , pp  ), (Ndprod,)  ,order='C').astype(numpy.int32) ,]
        dt_indx = dt_indx + [ numpy.reshape(   numpy.roll(  mylist, -1 , pp  ) , (Ndprod,) ,order='C').astype(numpy.int32) ,]

    return d_indx,  dt_indx

def QR_process(om, N, J, K, sn):
    """
    1D QR method for generating min-max interpolator

    :param om: non-Cartesian coordinate. shape = (M, dims)
    :param N: length
    :param J: size of interpolator
    :param K: size of oversampled grid
    :param sn:  scaling factor as a length-N vector
    :type om: numpy.float32
    :type N: int
    :type J: int
    :type K: int
    :type sn: numpy.float32 shape = (N,)
    """

    M = numpy.size(om)  # 1D size
    gam = 2.0 * numpy.pi / (K * 1.0)
    nufft_offset0 = nufft_offset(om, J, K)  # om/gam -  nufft_offset , [M,1]
    dk = 1.0 * om / gam - nufft_offset0  # om/gam -  nufft_offset , [M,1]


    arg = outer_sum(-numpy.arange(1, J + 1) * 1.0, dk) #[J, M]
    C_0 =numpy.outer(numpy.arange(0, N) - N/2,  numpy.arange(1, J + 1))

    C = numpy.exp(1.0j * gam*C_0)


    sn2 = numpy.reshape(sn, (N, 1))
    C = C*sn2
    bn =numpy.exp(1.0j*gam* numpy.outer(numpy.arange(0, N) - N/2, dk))

    return C, arg, bn

def solve_c(C, bn):
    CH = C.T.conj()
    C = CH.dot(C)
    bn = CH.dot(bn)

    C = numpy.linalg.pinv(C)
    c = C.dot(bn)
    return c

def QR2(om, N, J, K, sn, ft_flag):
    C, arg, bn = QR_process(om, N, J, K, sn)
    c = solve_c(C, bn)
    u2 = OMEGA_u(c, N, K, om, arg, ft_flag).T.conj()

    return u2



def get_sn(J, K, N):
    """
    Compute the 1D scaling factor for the given J, K, N

    :param J: size of interpolator
    :param K: size of oversampled grid
    :param N: length
    :type J: int
    :type N: int
    :type K: int
    :return: sn:  scaling factor as a length-N vector
    :rtype sn: numpy.float32 shape = (N,)
    """

    Nmid = (N - 1.0) / 2.0
    nlist = numpy.arange(0, N) * 1.0 - Nmid
    (kb_a, kb_m) = kaiser_bessel('string', J, 'best', 0, K / N)
    if J > 1:
        sn = 1.0/ kaiser_bessel_ft(nlist / K, J, kb_a, kb_m, 1.0)
    #                 sn = 1.0 / kaiser_bessel_ft(nlist / K, J, kb_a, kb_m, 1.0)
    elif J == 1:  # The case when samples are on regular grids
        sn = numpy.ones((1, N), dtype=dtype)
    return sn

def OMEGA_u(c, N, K, omd, arg, ft_flag):

    gam = 2.0 * numpy.pi / (K * 1.0)

    phase_scale =  1.0j * gam * (N*1.0 - 1.0) / 2.0
    phase = numpy.exp(phase_scale * arg)  # [J? M] linear phase

    if ft_flag is True:
        u = phase * c
        phase0= numpy.exp( - 1.0j*omd*N/2.0)
        u = phase0 * u

    else:
        u = c
    return u

def OMEGA_k(J,K, omd, Kd, dimid, dd, ft_flag):
    """
    Compute the index of k-space k_indx
    """
        # indices into oversampled FFT components
    # FORMULA 7
    M = omd.shape[0]
    koff = nufft_offset(omd, J, K)
    # FORMULA 9, find the indexes on Kd grids, of each M point
    if ft_flag is True: # tensor
        k_indx = numpy.mod(
            outer_sum(
                numpy.arange(
                    1,
                    J + 1) * 1.0,
                koff),
            K)
    else:
        k_indx =  numpy.reshape(omd, (1, M)).astype(numpy.int)
    #         k_indx = numpy.mod(
    #             outer_sum(
    #                 numpy.arange(
    #                     1,
    #                     J + 1) * 1.0,
    #                 koff),
    #             K)

    """
        JML: For GPU computing, indexing must be C-order (row-major)
        Multi-dimensional cuda or opencl arrays are row-major (order="C"), which  starts from the higher dimension.
        Note: This is different from the MATLAB indexing(for fortran order, column major, low-dimension first
    """

    if dimid < dd - 1:  # trick: pre-convert these indices into offsets!
        #            ('trick: pre-convert these indices into offsets!')
        k_indx = k_indx * numpy.prod(Kd[dimid+1:dd])# - 1
#     print(dimid, k_indx[0,0])
    """
    Note: F-order matrices must be reshaped into an 1D array before sparse matrix-vector multiplication.
    The original F-order (in Fessler and Sutton 2003) is not suitable for GPU array (C-order).
    Currently, in-place reshaping in F-order only works in numpy.

    """
    #             if dimid > 0:  # trick: pre-convert these indices into offsets!
    #                 #            ('trick: pre-convert these indices into offsets!')
    #                 kd[dimid] = kd[dimid] * numpy.prod(Kd[0:dimid]) - 1
    return k_indx

class pELL:
    """
    class pELL: partial ELL format
    """
    def __init__(self, M,  Jd, curr_sumJd, meshindex, kindx, udata):
        """
        Constructor

        :param M: Number of samples
        :type M: int
        :param Jd: Interpolator size
        :type Jd: tuple of int
        :param curr_sumJd: Summation of Jd[0:d-1], for fast shift computing
        :type curr_sumJd: tuple of int
        :param meshindex: The tensor indices to all interpolation points
        :type meshindex: numpy.uint32, shape =  (numpy.prod(Jd),  dd)
        :param kindx: Premixed k-indices to be combined
        :type kindx: numpy.uint32, shape = (M, numpy.sum(Jd))
        :param udata: Premixed interpolation data values
        :type udata: numpy.complex64, shape = (M, numpy.sum(Jd))
        :returns: pELL: partial ELLpack class with the given values
        :rtype: pELL: partial ELLpack class

        """
        self.nRow = M
        self.prodJd = numpy.prod(Jd)
        self.dim = len(Jd)
        self.sumJd = numpy.sum(Jd)
        self.Jd =  numpy.array(Jd).astype(numpy.uint32)
        self.curr_sumJd = curr_sumJd
        self.meshindex = numpy.array(meshindex, order='C')
        self.kindx = numpy.array(kindx, order='C')
        self.udata = udata.astype(numpy.complex64)

class Tensor_sn:
    '''
    Not implemented:
    '''
    def __init__(self, snd, radix):
#         raise NotImplementedError
        self.radix = radix
        Ndims = len(snd)
        Nd = ()
        for n in range(0, Ndims):
            Nd += (snd[n].shape[0], )



        Tdims = int(numpy.ceil(Ndims / radix)) # the final tensor dimension


        self.Tdims = Tdims
        Td = ()
        snd2 = ()
        for count in range(0, Tdims):
            d_start = count*radix
            d_end = (count + 1)*radix
            if d_end > Ndims:
                d_end = Ndims
            Td += (numpy.prod(Nd[d_start:d_end]), )


            Tsn = kronecker_scale(snd[d_start:d_end]).real.flatten()
            snd2 += (Tsn.reshape((Tsn.shape[0],1)), )

        tensor_sn = cat_snd(snd2) # Borrow the 1D method to concatenate the radix snds
        self.Td = Td
#         print('Td = ', Td)
        self.Td_elements, self.invTd_elements = strides_divide_itemsize(Td)

        self.tensor_sn = tensor_sn.astype(numpy.float32)


def create_csr(uu, kk, Kd, Jd, M):
#     Jprod = numpy.prod(Jd)
#     mm = numpy.arange(0, M).reshape( (1, M), order='C')  # indices from 0 to M-1
#     mm_rep = numpy.ones(( Jprod, 1))
#     mm = mm * mm_rep
    #     Jprod = numpy.prod(Jd)
    csrdata =uu.ravel(order='C')#numpy.reshape(uu.T, (Jprod * M, ), order='C')

    # row indices, from 1 to M convert array to list
#     rowindx = mm.ravel(order='C') #numpy.reshape(mm.T, (Jprod * M, ), order='C')

    Jdprod = numpy.prod(Jd)
    rowptr = numpy.arange(0, (M+1)*Jdprod, Jdprod)
    # colume indices, from 1 to prod(Kd), convert array to list
    colindx =kk.ravel(order='C')#numpy.reshape(kk.T, (Jprod * M, ), order='C')

    # The shape of sparse matrix
    csrshape = (M, numpy.prod(Kd))

    # Build sparse matrix (interpolator)
#     csr = scipy.sparse.csr_matrix((csrdata, (rowindx, colindx)),
#                                        shape=csrshape)
    csr = scipy.sparse.csr_matrix((csrdata, colindx, rowptr),
                                       shape=csrshape)
#     csr.has_sorted_indices = False
#     csr.sort_indices() # sort the indices in-place
    return csr
class ELL:
    """
    ELL is slow on a single core CPU
    """
    def __init__(self, elldata, ellcol):

#         self.shape = shape
        self.colWidth = ellcol.shape[1]
        self.nRow = ellcol.shape[0]
        self.data = elldata.reshape((self.nRow, self.colWidth),order='C')
        self.col = ellcol.astype(numpy.int32).reshape((self.nRow, self.colWidth),order='C')
    def spmv(self, x):
        y = numpy.einsum('ij,ij->i', self.data, x[self.col])
#         y = self.data * x[self.col]
#         y = numpy.sum(y, 1)
        return y

    def spmvH(self, y):
        x = numpy.zeros(self.shape[1],  dtype = numpy.complex64)
        x[self.col.ravel()] += numpy.einsum('ij,i->ij', self.data.conj(), y).ravel()
        return x
def create_ell(uu, kk):
#     Jprod = numpy.prod(Jd)

    # The shape of sparse matrix
#     ellshape = (M, numpy.prod(Kd))

    # Build sparse matrix (interpolator)
#     csr = scipy.sparse.csr_matrix((csrdata, (rowindx, colindx)),
#                                        shape=csrshape)
    ell = ELL(uu, kk)
#     csr.has_sorted_indices = False
#     csr.sort_indices() # sort the indices in-place
    return ell
def create_partialELL(ud, kd, Jd, M):
    """

    :param ud: tuple of all 1D interpolators
    :param kd: tuple of all 1D indices
    :param Jd: tuple of interpolation sizes
    :param M: number of samples
    :type ud: tuple of numpy.complex64 arrays
    :type kd: tuple of numpy.int32 arrays
    :type Jd: tuple of int32
    :type M: int
    :return: partialELL:
    :rtype: partialELL: pELL instance
    """
    dd = len(Jd)
    curr_sumJd = numpy.zeros( ( dd, ), dtype = numpy.uint32)
    kindx = numpy.zeros( ( M, numpy.sum(Jd)), dtype = numpy.uint32)
    udata = numpy.zeros( ( M, numpy.sum(Jd)), dtype = numpy.complex128)

    meshindex = numpy.zeros(  (numpy.prod(Jd),  dd), dtype = numpy.uint32)

    tmp_curr_sumJd = 0
    for dimid in range(0, dd):
        J = Jd[dimid]
        curr_sumJd[dimid] = tmp_curr_sumJd
        tmp_curr_sumJd +=int( J) # for next loop
        kindx[:, int(curr_sumJd[dimid] ): int(curr_sumJd[dimid]  + J)] = numpy.array(kd[dimid], order='C')
        udata[:, int(curr_sumJd[dimid] ): int(curr_sumJd[dimid]  + J)] = numpy.array(ud[dimid], order='C')

    series_prodJd = numpy.arange(0, numpy.prod(Jd))
    del ud, kd
    for dimid in range(dd-1, -1, -1):  # iterate over all dimensions

        J = Jd[dimid]
        xx = series_prodJd % J
        yy = numpy.floor(series_prodJd/ J)
        series_prodJd =  yy
        meshindex[:, dimid] = xx.astype(numpy.uint32)
#         else:
#             meshindex[:, dimid] = yy.astype(numpy.uint32)
#     print('dd=', dd)
#     print('Jd=', Jd)
#     print('curr_sumJd', curr_sumJd)
#     print('meshindex,', meshindex)
#     print('kindx.shape = ', kindx.shape)
#     print('udata.shape = ', udata.shape)
    partialELL = pELL(M, Jd, curr_sumJd, meshindex, kindx, udata.astype(dtype))
    return partialELL

# def partial_combination(ud, kd, Jd):
#     """
#     Input:
#     ud (the struct of all 1D interpolators), kd (the struct of all 1D indeces of 1D interpolators),
#     Jd: tuple of interpolation sizes
#     dd: the number of dimensions
#     M: the number of samples
#
#     output:
#     M: number of non-uniform locations
#     Jd: tuple,  (Jd[0], Jd[1], Jd[2],    ...,    Jd[dd -1])
#     curr_sumJd: summation of curr_sumJd[dimid] = numpy.sum(Jd[0:dimid - 1])
#     meshindex: For prodJd hypercubic interpolators, find the indices of tensor, shape = (prodJd, dd)
#     kindx: column indeces, shape = (M, sumJd)
#     udata: interpolators, shape = (M, sumJd)
#
#     """
#
#     dd = len(Jd)
#
#     kk = kd[0]  # [M, J1] # pointers to indices
#     M = kd[0].shape[0]
#     uu = ud[0]  # [M, J1]
#     Jprod = Jd[0]
#     for dimid in range(1, dd):
#         Jprod *= Jd[dimid]#numpy.prod(Jd[:dimid + 1])
#
#         kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
#         kk = kk.reshape((M, Jprod), order='C')
#         uu = numpy.einsum('ij,ik->ijk', uu, ud[dimid])
#         uu = uu.reshape((M, Jprod), order='C')
#     kd2 = (kk, )
#     ud2 = (uu, )
#     Jd2 = (Jprod, )
#     return ud2, kd2, Jd2
def rdx_N(ud, kd, Jd):
    ud2 = (khatri_rao_u(ud), )
    kd2 = (khatri_rao_k(kd), )
    Jd2 = (numpy.prod(Jd), )

    return ud2, kd2, Jd2
def full_kron(ud, kd, Jd, Kd, M):
#     (udata, kindx)=khatri_rao(ud, kd, Jd)

#     udata = khatri_rao_u(ud)
#     kindx = khatri_rao_k(kd)
    ud2, kd2, Jd2 = rdx_N(ud, kd, Jd)
    CSR  = create_csr(ud2[0], kd2[0], Kd, Jd, M) # must have
    # Dimension reduction: Nd -> 1
    # Tuple (Nd) -> array (shape = M*prodJd)

#     Note: the shape of uu and kk is (M, prodJd)
#     ELL = create_ell(   udata,  kindx)#, Kd, Jd, M)
    return CSR#, ELL
def khatri_rao_k(kd):
    dd = len(kd)

    kk = kd[0]  # [M, J1] # pointers to indices
    M = kd[0].shape[0]
#     uu = ud[0]  # [M, J1]
    Jprod = kd[0].shape[1]
    for dimid in range(1, dd):
        Jprod *= kd[dimid].shape[1] #numpy.prod(Jd[:dimid + 1])

        kk = block_outer_sum(kk, kd[dimid]) #+ 1  # outer sum of indices
        kk = kk.reshape((M, Jprod), order='C')
#         uu = numpy.einsum('mi,mj->mij', uu, ud[dimid])
#         uu = uu.reshape((M, Jprod), order='C')

    return kk
def khatri_rao_u( ud):
    dd = len(ud)

#     kk = kd[0]  # [M, J1] # pointers to indices
    M = ud[0].shape[0]
    uu = ud[0]  # [M, J1]
    Jprod = ud[0].shape[1]
    for dimid in range(1, dd):
        Jprod *=ud[dimid].shape[1]#numpy.prod(Jd[:dimid + 1])

#         kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
#         kk = kk.reshape((M, Jprod), order='C')
        uu = numpy.einsum('mi,mj->mij', uu, ud[dimid])
        uu = uu.reshape((M, Jprod), order='C')

    return uu
# def khatri_rao(ud, kd, Jd):
#     dd = len(Jd)
#
#     kk = kd[0]  # [M, J1] # pointers to indices
#     M = kd[0].shape[0]
#     uu = ud[0]  # [M, J1]
#     Jprod = Jd[0]
#     for dimid in range(1, dd):
#         Jprod *= Jd[dimid]#numpy.prod(Jd[:dimid + 1])
#
#         kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
#         kk = kk.reshape((M, Jprod), order='C')
#         uu = numpy.einsum('mi,mj->mij', uu, ud[dimid])
#         uu = uu.reshape((M, Jprod), order='C')
#
#     return uu, kk
def rdx_kron(ud, kd, Jd, radix=None):
    """
    Radix-n Kronecker product of multi-dimensional array

    :param ud: 1D interpolators
    :type ud: tuple of (M, Jd[d]) numpy.complex64 arrays
    :param kd: 1D indices to interpolators
    :type kd: tuple of (M, Jd[d]) numpy.uint arrays
    :param Jd: 1D interpolator sizes
    :type Jd: tuple of int
    :param radix: radix of Kronecker product
    :type radix: int
    :returns: uu: 1D interpolators
    :type uu: tuple of (M, Jd[d]) numpy.complex64 arrays
    :param kk: 1D indices to interpolators
    :type kk: tuple of (M, Jd[d]) numpy.uint arrays
    :param JJ: 1D interpolator sizes
    :type JJ: tuple of int

    """
    M = ud[0].shape[0]
    dd = len(Jd)
    if radix is None:
        radix = dd
    if radix > dd:
        radix = dd

    ud2 = ()
    kd2 = ()
    Jd2 = ()
#     kk = kd[0]  # [J1 M] # pointers to indices
#     uu = ud[0]  # [J1 M]
#     Jprod = Jd[0]
    for count in range(0, int(numpy.ceil(dd/radix)), ):
#         kk = kd[count]  # [J1 M] # pointers to indices
#         uu = ud[count]  # [J1 M]
#         Jprod = Jd[count]
#         uu = numpy.ones((M, 1), dtype = numpy.complex64)
#         Jprod = 1
        d_start = count*radix
        d_end = (count + 1)*radix
        if d_end > dd:
            d_end = dd
        ud3, kd3, Jd3 = rdx_N(ud[d_start:d_end], kd[d_start:d_end], Jd[d_start:d_end])
        ud2 += ud3
        kd2 += kd3
        Jd2 += Jd3
#         for dimid in range(d_start + 1, d_end):
#             Jprod *= Jd[dimid]#numpy.prod(Jd[:dimid + 1])
#
#             kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
#             kk = kk.reshape((M, Jprod), order='C')
#             uu = numpy.einsum('ij,ik->ijk', uu, ud[dimid])
#             uu = uu.reshape((M, Jprod), order='C')


#         ud2 += (uu, )
#         kd2 += (kk, )
#         Jd2 += (Jprod, )
    return ud2, kd2, Jd2 #(uu, ), (kk, ), (Jprod, )#, Jprod

def kronecker_scale(snd):
    """
    Compute the Kronecker product of the scaling factor.

    :param snd: Tuple of 1D scaling factors
    :param dd: Number of dimensions
    :type snd: tuple of 1D numpy.array
    :return: sn: The multi-dimensional Kronecker of the scaling factors
    :rtype: Nd array

    """
    dd = len(snd)
    shape_broadcasting = ()
    for dimid in range(0, dd):
        shape_broadcasting += (1, )
#     sn = numpy.array(1.0 + 0.0j)
    sn= numpy.reshape(1.0, shape_broadcasting)
    for dimid in range(0, dd):
        sn_shape = list(shape_broadcasting)
        sn_shape[dimid] = snd[dimid].shape[0]
        tmp = numpy.reshape(snd[dimid], tuple(sn_shape))
#         print('tmp.shape = ', tmp.shape)
        ###############################################################
        # higher dimension implementation: multiply over all dimension
        ###############################################################
        sn = sn  * tmp # multiply using broadcasting
    return sn

def cat_snd(snd):
    """
    :param snd:  tuple of input 1D vectors
    :type snd: tuple
    :return:  tensor_sn: vector of concatenated scaling factor, shape = (numpy.sum(Nd), )
    :rtype: tensor_sn: numpy.float32
    """
    Nd = ()
    dd = len(snd)
    for dimid in range(0, dd):
        Nd += (snd[dimid].shape[0],)
    tensor_sn = numpy.empty((numpy.sum(Nd), ), dtype=numpy.float64)

    shift = 0
    for dimid in range(0, len(Nd)):

        tensor_sn[shift :shift + Nd[dimid]] = snd[dimid][:,0].real
        shift = shift + Nd[dimid]
    return tensor_sn

def min_max(N, J, K, alpha, beta, om, ft_flag):
    T = nufft_T(    N,  J,  K,  alpha,  beta)
    ###############################################################
    # formula 30  of Fessler's paper
    ###############################################################
    (r, arg) = nufft_r( om,   N,  J,
                   K,   alpha,  beta)  # large N approx [J? M]
    ###############################################################
    # Min-max interpolator
    ###############################################################
    c = T.dot(r)
    u2 = OMEGA_u(c, N, K, om, arg, ft_flag).T.conj()
    return u2

def plan(om, Nd, Kd, Jd, ft_axes = None, format='CSR', radix = None):
    """
    Plan for the NUFFT object.

    :param om: Coordinate
    :param Nd: Image shape
    :param Kd: Oversampled grid shape
    :param Jd: Interpolator size
    :param ft_axes: Axes where FFT takes place
    :param format: Output format of the interpolator.
                    'CSR': the precomputed Compressed Sparse Row (CSR) matrix.
                    'pELL': partial ELLPACK which precomputes the concatenated 1D interpolators.
    :type om: numpy.float
    :type Nd: tuple of int
    :type Kd: tuple of int
    :type Jd: tuple of int
    :type ft_axes: tuple of int
    :type format: string, 'CSR' or 'pELL'
    :return st: dictionary for NUFFT

    """

#         self.debug = 0  # debug

    if type(Nd) != tuple:
        raise TypeError('Nd must be tuple, e.g. (256, 256)')

    if type(Kd) != tuple:
        raise TypeError('Kd must be tuple, e.g. (512, 512)')

    if type(Jd) != tuple:
        raise TypeError('Jd must be tuple, e.g. (6, 6)')

    if (len(Nd) != len(Kd)) | (len(Nd) != len(Jd))  | len(Kd) != len(Jd):
        raise KeyError('Nd, Kd, Jd must be in the same length, e.g. Nd=(256,256),Kd=(512,512),Jd=(6,6)')

    dd = numpy.size(Nd)

    if ft_axes is None:
        ft_axes = tuple(xx for xx in range(0, dd))

#     print('ft_axes = ', ft_axes)
    ft_flag = () # tensor

    for pp in range(0, dd):
        if pp in ft_axes:
            ft_flag += (True, )
        else:
            ft_flag += (False, )
#     print('ft_flag = ', ft_flag)
###############################################################
# check input errors
###############################################################
    st = {}


###############################################################
# First, get alpha and beta: the weighting and freq
# of formula (28) in Fessler's paper
# in order to create slow-varying image space scaling
###############################################################
#     for dimid in range(0, dd):
#         (tmp_alpha, tmp_beta) = nufft_alpha_kb_fit(
#             Nd[dimid], Jd[dimid], Kd[dimid])
#         st.setdefault('alpha', []).append(tmp_alpha)
#         st.setdefault('beta', []).append(tmp_beta)
    st['tol'] = 0
    st['Jd'] = Jd
    st['Nd'] = Nd
    st['Kd'] = Kd
    M = om.shape[0]
    st['M'] = numpy.int32(M)
    st['om'] = om

###############################################################
# create scaling factors st['sn'] given alpha/beta
# higher dimension implementation
###############################################################

    """
    Now compute the 1D scaling factors
    snd: list
    """

    for dimid in range(0, dd):

        (tmp_alpha, tmp_beta) = nufft_alpha_kb_fit(
            Nd[dimid], Jd[dimid], Kd[dimid])
        st.setdefault('alpha', []).append(tmp_alpha)
        st.setdefault('beta', []).append(tmp_beta)

    snd = []
    for dimid in range(0, dd):
        snd += [nufft_scale(
            Nd[dimid],
            Kd[dimid],
            st['alpha'][dimid],
            st['beta'][dimid]), ]
    """
     higher-order Kronecker product of all dimensions
    """

    # [J? M] interpolation coefficient vectors.
    # Iterate over all dimensions and
    # multiply the coefficients of all dimensions

    ud = []
    for dimid in range(0, dd):  # iterate through all dimensions
        N = Nd[dimid]
        J = Jd[dimid]
        K = Kd[dimid]
        alpha = st['alpha'][dimid]
        beta = st['beta'][dimid]
       ###############################################################
        # formula 29 , 26 of Fessler's paper
        ###############################################################

        # pseudo-inverse of CSSC using large N approx [J? J?]
        if ft_flag[dimid] is True:

#         ###############################################################
#         # formula 30  of Fessler's paper
#         ###############################################################

#         ###############################################################
#         # fast approximation to min-max interpolator
#         ###############################################################

#             c, arg = min_max(N, J, K, alpha, beta, om[:, dimid])
#         ###############################################################
#        # QR: a more accurate solution but slower than above fast approximation
#        ###############################################################

#             c, arg = QR_process(om[:,dimid], N, J, K, snd[dimid])

            #### phase shift
#             ud += [QR2(om[:,dimid], N, J, K, snd[dimid], ft_flag[dimid]),]
            ud += [min_max(N, J, K, alpha, beta, om[:, dimid], ft_flag[dimid]),]

        else:
            ud += [numpy.ones((1, M), dtype = dtype).T, ]


    """
    Now compute the column indices for 1D interpolators
    Each length-Jd interpolator includes Jd points, which are linked to Jd k-space locations
    kd is a tuple storing the 1D interpolators.
    A following Kronecker product will be needed.
    """
    kd = []
    for dimid in range(0, dd):  # iterate over all dimensions

        kd += [OMEGA_k(Jd[dimid],Kd[dimid], om[:,dimid], Kd, dimid, dd, ft_flag[dimid]).T, ]


    if format is 'CSR':

        CSR = full_kron(ud, kd, Jd, Kd, M)
        st['p'] = CSR
#     st['ell'] = ELL
        st['sn'] = kronecker_scale(snd).real # only real scaling is relevant
        st['tSN'] = Tensor_sn(snd, len(Kd))
#     ud2, kd2, Jd2 = partial_combination(ud, kd, Jd)
    elif format is 'pELL':
        if radix is None:
            radix = 1
        ud2, kd2, Jd2 = rdx_kron(ud, kd, Jd, radix=radix)
#         print(ud2[0].shape, ud2[1].shape, kd2[0].shape, kd2[1].shape, Jd2)
        st['pELL'] = create_partialELL(ud2, kd2, Jd2, M)
#         st['tensor_sn'] = snd
#         st['tensor_sn'] = cat_snd(snd)
        st['tSN'] = Tensor_sn(snd, radix)
#         numpy.empty((numpy.sum(Nd), ), dtype=numpy.float32)
#
#         shift = 0
#         for dimid in range(0, len(Nd)):
#
#             st['tensor_sn'][shift :shift + Nd[dimid]] = snd[dimid][:,0].real
#             shift = shift + Nd[dimid]
    # no dimension-reduction Nd -> Nd
    # Tuple (Nd) -> array (shape = M*sumJd)

    return st #new

def strides_divide_itemsize(Nd):
    """
    strides_divide_itemsize function computes the step_size (strides/itemsize) along different axes, and its inverse as float32.
    For fast GPU computing, preindexing allows for fast Hadamard product and copy.
    However preindexing costs some memory.
    strides_divide_itemsize aims to replace preindexing by run-time calculation of the index, given the invNd_elements.

    :param Nd: Input shape
    :type Nd: tuple of int
    :return: Nd_elements: strides/itemsize of the Nd.
    :return:  invNd_elements: (float32)(1/Nd_elements). Division on GPU is slow but multiply is fast. Thus we can precompute the inverse and then multiply the inverse on GPU.
    :rtype: Nd_elements: tuple of int
    :rtype: invNd_elements: tuple of float32

    .. seealso:: :class:`pynufft.NUFFT_hsa`

    """

    Nd_elements = tuple(numpy.prod(Nd[dd+1:]) for dd in range(0,len(Nd)))
#     Kd_elements = tuple(numpy.prod(Kd[dd+1:]) for dd in range(0,len(Kd)))
    invNd_elements = 1/numpy.array(Nd_elements, dtype = numpy.float32)

    return Nd_elements, invNd_elements



def preindex_copy(Nd, Kd):
    """
    Building the array index for copying two arrays of sizes Nd and Kd.
    Only the front parts of the input/output arrays are copied.
    The oversize  parts of the input array are truncated (if Nd > Kd), 
    and the smaller size are zero-padded (if Nd < Kd)

    :param Nd: tuple, the dimensions of array1
    :param Kd: tuple, the dimensions of array2
    :type Nd: tuple with integer elements
    :type Kd: tuple with integer elements
    :return: inlist: the index of the input array
    :return: outlist: the index of the output array
    :return: nelem: the length of the inlist and outlist (equal length)
    :rtype: inlist: list with integer elements
    :rtype: outlist: list with integer elements
    :rtype: nelem: int
    """
    ndim = len(Nd)
    kdim = len(Kd)
    if ndim != kdim:
        print("mismatched dimensions!")
        print("Nd and Kd must have the same dimensions")
        raise
    else:
        nelem = 1
        min_dim = ()
        for pp in range(ndim - 1, -1,-1):
            YY = numpy.minimum(Nd[pp], Kd[pp])
            nelem *= YY
            min_dim = (YY,) + min_dim
        mylist = numpy.arange(0, nelem).astype(numpy.int32)
#             a=mylist
        BB=()
        for pp in range(ndim - 1, 0, -1):
             a = numpy.floor(mylist/min_dim[pp])
             b = mylist%min_dim[pp]
             mylist = a
             BB=(b,) + BB

        if ndim == 1:
            mylist =  numpy.arange(0, nelem).astype(numpy.int32)
        else:
            mylist = numpy.floor( numpy.arange(0, nelem).astype(numpy.int32)/ numpy.prod(min_dim[1:]))

        inlist = mylist
        outlist = mylist
        for pp in range(0, ndim-1):
            inlist = inlist*Nd[pp+1] + BB[pp]
            outlist = outlist*Kd[pp+1] + BB[pp]

    return inlist.astype(numpy.int32), outlist.astype(numpy.int32), nelem.astype(numpy.int32)

def dirichlet(x):
    return numpy.sinc(x)

def outer_sum(xx, yy):
    """
    Superseded by numpy.add.outer() function
    """

    return numpy.add.outer(xx,yy)
#     nx = numpy.size(xx)
#     ny = numpy.size(yy)
#
#     arg1 = numpy.tile(xx, (ny, 1)).T
#     arg2 = numpy.tile(yy, (nx, 1))
#     return arg1 + arg2


def nufft_offset(om, J, K):
    '''
    For every om point (outside regular grids), find the nearest
    central grid (from Kd dimension)
    '''
    gam = 2.0 * numpy.pi / (K * 1.0)
    k0 = numpy.floor(1.0 * om / gam - 1.0 * J / 2.0)  # new way
    return k0


def nufft_alpha_kb_fit(N, J, K):
    """
    Find parameters alpha and beta for scaling factor st['sn']
    The alpha is hardwired as [1,0,0...] when J = 1 (uniform scaling factor)

    :param N: size of image
    :param J: size of interpolator
    :param K: size of oversampled k-space
    :type N: int
    :type J: int
    :type K: int
    :returns: alphas:
    :returns: beta:
    :rtype: alphas: list of float
    :rtype: beta:
    """

    beta = 1
    Nmid = (N - 1.0) / 2.0
    if N > 40:
        L = 13
    else:
        L = numpy.ceil(N / 3).astype(numpy.int16)

    nlist = numpy.arange(0, N) * 1.0 - Nmid
    (kb_a, kb_m) = kaiser_bessel('string', J, 'best', 0, K / N)
    if J > 1:
        sn_kaiser = 1 / kaiser_bessel_ft(nlist / K, J, kb_a, kb_m, 1.0)
    elif J == 1:  # The case when samples are on regular grids
        sn_kaiser = numpy.ones((1, N), dtype=dtype)
    gam = 2 * numpy.pi / K
    X_ant = beta * gam * nlist.reshape((N, 1), order='F')
    X_post = numpy.arange(0, L + 1)
    X_post = X_post.reshape((1, L + 1), order='F')
    X = numpy.dot(X_ant, X_post)  # [N,L]
    X = numpy.cos(X)
    sn_kaiser = sn_kaiser.reshape((N, 1), order='F').conj()
    X = numpy.array(X, dtype=dtype)
    sn_kaiser = numpy.array(sn_kaiser, dtype=dtype)
    coef = numpy.linalg.lstsq(numpy.nan_to_num(X), numpy.nan_to_num(sn_kaiser), rcond = -1)[0]
    alphas = coef
    if J > 1:
        alphas[0] = alphas[0]
        alphas[1:] = alphas[1:] / 2.0
    elif J == 1:  # cases on grids
        alphas[0] = 1.0
        alphas[1:] = 0.0
    alphas = numpy.real(alphas)
    return (alphas, beta)


def kaiser_bessel(x, J, alpha, kb_m, K_N):
    if K_N != 2:
        kb_m = 0
        alpha = 2.34 * J
    else:
        kb_m = 0

        # Parameters in Fessler's code
        # because it was experimentally determined to be the best!
        # input: number of interpolation points
        # output: Kaiser_bessel parameter

        jlist_bestzn = {2: 2.5,
                        3: 2.27,
                        4: 2.31,
                        5: 2.34,
                        6: 2.32,
                        7: 2.32,
                        8: 2.35,
                        9: 2.34,
                        10: 2.34,
                        11: 2.35,
                        12: 2.34,
                        13: 2.35,
                        14: 2.35,
                        15: 2.35,
                        16: 2.33}

        if J in jlist_bestzn:
            alpha = J * jlist_bestzn[J]
        else:
            tmp_key = (jlist_bestzn.keys())
            min_ind = numpy.argmin(abs(tmp_key - J * numpy.ones(len(tmp_key))))
            p_J = tmp_key[min_ind]
            alpha = J * jlist_bestzn[p_J]
    kb_a = alpha
    return (kb_a, kb_m)


def kaiser_bessel_ft(u, J, alpha, kb_m, d):
    '''
    Interpolation weight for given J/alpha/kb-m
    '''

    u = u * (1.0 + 0.0j)
    import scipy.special
    z = numpy.sqrt((2 * numpy.pi * (J / 2) * u) ** 2.0 - alpha ** 2.0)
    nu = d / 2 + kb_m
    y = ((2 * numpy.pi) ** (d / 2)) * ((J / 2) ** d) * (alpha ** kb_m) / \
        scipy.special.iv(kb_m, alpha) * scipy.special.jv(nu, z) / (z ** nu)
    y = numpy.real(y)
    return y


def nufft_scale1(N, K, alpha, beta, Nmid):
    '''
    Calculate image space scaling factor
    '''
#     import types
#     if alpha is types.ComplexType:
    alpha = numpy.real(alpha)
#         print('complex alpha may not work, but I just let it as')

    L = len(alpha) - 1
    if L > 0:
        sn = numpy.zeros((N, 1))
        n = numpy.arange(0, N).reshape((N, 1), order='F')
        i_gam_n_n0 = 1j * (2 * numpy.pi / K) * (n - Nmid) * beta
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)]
            if l1 < 0:
                alf = numpy.conj(alf)
            sn = sn + alf * numpy.exp(i_gam_n_n0 * l1)
    else:
        sn = numpy.dot(alpha, numpy.ones((N, 1)))
    return sn


def nufft_scale(Nd, Kd, alpha, beta):
    dd = numpy.size(Nd)
    Nmid = (Nd - 1) / 2.0
    if dd == 1:
        sn = nufft_scale1(Nd, Kd, alpha, beta, Nmid)
    else:
        sn = 1
        for dimid in numpy.arange(0, dd):
            tmp = nufft_scale1(Nd[dimid], Kd[dimid], alpha[dimid],
                               beta[dimid], Nmid[dimid])
            sn = numpy.dot(list(sn), tmp.H)
    return sn


def mat_inv(A):
#     I = numpy.eye(A.shape[0], A.shape[1])
    # B = scipy.linalg.pinv2(A)
    B = scipy.linalg.pinv(A)
    return B


def nufft_T(N, J, K, alpha, beta):
    '''
     Equation (29) and (26) in Fessler and Sutton 2003.
     Create the overlapping matrix CSSC (diagonal dominant matrix)
     of J points, then find the pseudo-inverse of CSSC '''

#     import scipy.linalg
    L = numpy.size(alpha) - 1
#     print('L = ', L, 'J = ',J, 'a b', alpha,beta )
    cssc = numpy.zeros((J, J))
    [j1, j2] = numpy.mgrid[1:J + 1, 1:J + 1]
    overlapping_mat = j2 - j1
    for l1 in range(-L, L + 1):
        for l2 in range(-L, L + 1):
            alf1 = alpha[abs(l1)]
#             if l1 < 0: alf1 = numpy.conj(alf1)
            alf2 = alpha[abs(l2)]
#             if l2 < 0: alf2 = numpy.conj(alf2)
            tmp = overlapping_mat + beta * (l1 - l2)

            tmp = dirichlet(1.0 * tmp / (1.0 * K / N))
            cssc = cssc + alf1 * alf2 * tmp

    return mat_inv(cssc)


def nufft_r(om, N, J, K, alpha, beta):
    '''
    Equation (30) of Fessler & Sutton's paper

    '''
    def iterate_sum(rr, alf, r1):
        rr = rr + alf * r1
        return rr
    def iterate_l1(L, alpha, arg, beta, K, N, rr):
        oversample_ratio = (1.0 * K / N)
        import time
        t0=time.time()
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)] * 1.0
    #         if l1 < 0:
    #             alf = numpy.conj(alf)
        #             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
            input_array = (arg + 1.0 * l1 * beta) / oversample_ratio
            r1 = dirichlet(input_array)
            rr = iterate_sum(rr, alf, r1)
        return rr

    M = numpy.size(om)  # 1D size
    gam = 2.0 * numpy.pi / (K * 1.0)
    nufft_offset0 = nufft_offset(om, J, K)  # om/gam -  nufft_offset , [M,1]
    dk = 1.0 * om / gam - nufft_offset0  # om/gam -  nufft_offset , [M,1]
    arg = outer_sum(-numpy.arange(1, J + 1) * 1.0, dk)
    L = numpy.size(alpha) - 1
#     print('alpha',alpha)
    rr = numpy.zeros((J, M), dtype=numpy.float32)
    rr = iterate_l1(L, alpha, arg, beta, K, N, rr)
    return (rr, arg)


def block_outer_sum0(x1, x2):
    '''
    Multiply x1 (J1 x M) and x2 (J2xM) and extend the dimensions to 3D (J1xJ2xM)
    '''
    (J1, M) = x1.shape
    (J2, M) = x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1, 1, M), order='C')  # [J1 1 M] from [J1 M]
#     xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='C')  # [1 J2 M] from [J2 M]
#     xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid

    y = xx1 + xx2
#     y = numpy.einsum('ik, jk->ijk', x1, x2)
    return y  # [J1 J2 M]

def block_outer_prod(x1, x2):
    '''
    Multiply x1 (J1 x M) and x2 (J2xM) and extend the dimensions to 3D (J1xJ2xM)
    '''
    (J1, M) = x1.shape
    (J2, M) = x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1, 1, M), order='C')  # [J1 1 M] from [J1 M]
#     xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='C')  # [1 J2 M] from [J2 M]
#     xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid

    y = xx1 * xx2
#     y = numpy.einsum('ik, jk->ijk', x1, x2)
    return y  # [J1 J2 M]


def block_outer_sum(x1, x2):
    '''
    Update the new index after adding a new axis
    '''
    (M, J1) = x1.shape
    (M, J2) = x2.shape
    xx1 = x1.reshape((M, J1, 1), order='C')  # [J1 1 M] from [J1 M]
    xx2 = x2.reshape((M, 1, J2), order='C')  # [1 J2 M] from [J2 M]
    y = xx1 + xx2
    return y  # [J1 J2 M]


def crop_slice_ind(Nd):
    '''
    (Deprecated in v.0.3.4)
    Return the "slice" of Nd size to index multi-dimensional array.  "Slice" functions as the index of the array.
    This function is superseded by preindex_copy(), which avoid run-time indexing.
    '''
    return [slice(0, Nd[ss]) for ss in range(0, len(Nd))]
def device_list():
    """
    device_list() returns available devices for acceleration as a tuple.
    If no device is available, it returns an empty tuple. 
    """
    from reikna import cluda
    import reikna.transformations
    from reikna.cluda import functions, dtypes
    devices_list = ()
    try:
        api = cluda.cuda_api()
        available_cuda_device = cluda.find_devices(api)
        cuda_flag = 1
        for api_n in available_cuda_device.keys():
            cuda_gpus = available_cuda_device[api_n]
    #         print('cuda_gpus = ', cuda_gpus)
            for dev_num in cuda_gpus:
                
                id = available_cuda_device[api_n][dev_num]
                platform = api.get_platforms()[api_n]
                device = platform.get_devices()[id]
                thr = api.Thread(device)
                wavefront = api.DeviceParameters(device).warp_size
                devices_list += (('cuda',  api_n, dev_num, platform, device, thr, wavefront),)
    except:
        print('No cuda device found. Check your pycuda installation.')
    try:
        api = cluda.ocl_api()
        available_ocl_device = cluda.find_devices(api)
        ocl_flag = 1
    #         if verbosity > 0:
    #             print("try to load cuda interface:")
    #             print(api, available_cuda_device)
    #     print(available_cuda_device.keys())
        for api_n in available_ocl_device.keys():
            ocl_gpus = available_ocl_device[api_n]
    #         print('cuda_gpus = ', cuda_gpus)
            for dev_num in ocl_gpus:
                
                id = available_ocl_device[api_n][dev_num]
                platform = api.get_platforms()[api_n]
                device = platform.get_devices()[id]
                thr = api.Thread(device)
                wavefront = api.DeviceParameters(device).warp_size
                devices_list += (('ocl',  api_n, dev_num, platform, device, thr, wavefront),)
    except:
        print('No OpenCL device found. Check your pyopencl installation.')
#             print("API='cuda',  ", "platform_number=", api_n,
#                   ", device_number=", available_cuda_device[api_n][0])
#     except:
#         pass
    
    return devices_list

def diagnose(verbosity=0):
    """
    Diagnosis function
    Find available devices when NUFFT.offload() fails.
    """
    from reikna import cluda
    import reikna.transformations
    from reikna.cluda import functions, dtypes
    try:
        api = cluda.cuda_api()
        if verbosity > 0:
            print('cuda interface is available')
        available_cuda_device = cluda.find_devices(api)
        cuda_flag = 1
        if verbosity > 0:
            print("try to load cuda interface:")
            print(api, available_cuda_device)
            for api_n in available_cuda_device.keys():
                print("API='cuda',  ", "platform_number=", api_n,
                      ", device_number=", available_cuda_device[api_n][0])
    except:
        cuda_flag = 0
        if verbosity > 0:
            print('cuda interface is not available')
    try:
        api = cluda.ocl_api()
        available_ocl_device = cluda.find_devices(api)
        ocl_flag = 1
        if verbosity > 0:
            print('ocl interface is available')
            print(api, available_ocl_device)
            print("try to load ocl interface with:")
            for api_n in available_ocl_device.keys():
                print("API='ocl',  ", "platform_number=", api_n,
                      ", device_number=", available_ocl_device[api_n][0])
    except:
        print('ocl interface is not available')
        ocl_flag = 0
    return cuda_flag, ocl_flag
if __name__ == '__main__':
    devices = device_list()
    for pp in range(0, len(devices)):
        print(devices[pp])
