"""
NUFFT HSA classes (deprecated)
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

from ..src._helper import helper, helper1


class hypercube:
    def __init__(self, shape, steps, invsteps, nelements, batch, dtype):
        self.shape = shape
        self.steps = steps
        self.invsteps = invsteps
        self.nelements = nelements
        self.batch = batch
        self.dtype = dtype


def push_cuda_context(hsa_method):
    """
    Decorator to push up CUDA context to the top of the stack for current use
    Add @push_cuda_context before the methods of NUFFT_hsa()
    """
    @_wraps(hsa_method)
    def wrapper(*args, **kwargs):
        try:
            args[0].thr._context.push()
        except:
            pass
        return hsa_method(*args, **kwargs)
    return wrapper


class NUFFT_hsa:
    """
    NUFFT_hsa class.
    Multi-coil or single-coil memory reduced NUFFT.

    """
    def __init__(self, API=None, platform_number=None, device_number=None,
                 verbosity=0):
        """
        Constructor.

        :param API: The API for the heterogeneous system. API='cuda'
                    or API='ocl'
        :param platform_number: The number of the platform found by the API.
        :param device_number: The number of the device found on the platform.
        :param verbosity: Defines the verbosity level, default value is 0
        :type API: string
        :type platform_number: integer
        :type device_number: integer
        :type verbosity: integer
        :returns: 0
        :rtype: int, float

        :Example:

        >>> from pynufft import NUFFT_hsa
        >>> NufftObj = NUFFT_hsa(API='cuda', platform_number=0,
                                         device_number=0, verbosity=0)
        """
        warnings.warn('In the future NUFFT_hsa and NUFFT_cpu api will'
                      ' be merged', FutureWarning)
        self.dtype = numpy.complex64
        self.verbosity = verbosity

        import reikna.cluda as cluda
        if self.verbosity > 0:
            print('The choosen API by the user is ', API)
        self.cuda_flag, self.ocl_flag = helper.diagnose(
            verbosity=self.verbosity)
        if None is API:
            if self.cuda_flag is 1:
                API = 'cuda'
            elif self.ocl_flag is 1:
                API = 'ocl'
            else:
                warnings.warn('No parallelization will be made since no GPU '
                              'device has been detected.', UserWarning)
        else:
            api = API
        if self.verbosity > 0:
            print('The used API will be ', API)
        if platform_number is None:
            platform_number = 0
        if device_number is None:
            device_number = 0

        from reikna import cluda
        import reikna.transformations
        from reikna.cluda import functions, dtypes
        try:  # try to create api/platform/device using the given parameters
            if 'cuda' == API:
                api = cluda.cuda_api()
            elif 'ocl' == API:
                api = cluda.ocl_api()

            platform = api.get_platforms()[platform_number]

            device = platform.get_devices()[device_number]
        except:  # if failed, find out what's going wrong?
            warnings.warn('No parallelization will be made since no GPU '
                          'device has been detected.', UserWarning)

#             return 1

#         Create context from device
        self.thr = api.Thread(device)  # pyopencl.create_some_context()
        self.device = device  # : device name
        if self.verbosity > 0:
            print('Using opencl or cuda = ', self.thr.api)

#         """
#         Wavefront: as warp in cuda. Can control the width in a workgroup
#         Wavefront is required in spmv_vector as it improves data coalescence.
#         see cCSR_spmv and zSparseMatVec
#         """
        self.wavefront = api.DeviceParameters(device).warp_size
        if self.verbosity > 0:
            print('Wavefront of OpenCL (as wrap of CUDA) = ', self.wavefront)

        from ..src import re_subroutine  # import create_kernel_sets
        kernel_sets = re_subroutine.create_kernel_sets(API)

        prg = self.thr.compile(kernel_sets,
                               render_kwds=dict(LL=str(self.wavefront)),
                               fast_math=False)
        self.prg = prg
    def set_wavefront(self, wf):
        self.wavefront = int(wt)#api.DeviceParameters(device).warp_size
        if self.verbosity > 0:
            print('Wavefront of OpenCL (as wrap of CUDA) = ', self.wavefront)

        from ..src import re_subroutine  # import create_kernel_sets
        kernel_sets = re_subroutine.create_kernel_sets(API)

        prg = self.thr.compile(kernel_sets,
                               render_kwds=dict(LL=str(self.wavefront)),
                               fast_math=False)
        self.prg = prg
    def plan(self, om, Nd, Kd, Jd, ft_axes=None, batch=None, radix=None):
        """
        Design the multi-coil or single-coil memory reduced interpolator.


        :param om: The M off-grid locations in the frequency domain.
                   Normalized between [-pi, pi]
        :param Nd: The matrix size of equispaced image.
                    Example: Nd=(256, 256) for a 2D image;
                             Nd = (128, 128, 128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid.
                   Example: Kd=(512,512) for 2D image;
                   Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size.
                   Example: Jd=(6,6) for 2D image;
                   Jd = (6,6,6) for a 3D image
        :param ft_axes: The dimensions to be transformed by FFT.
                   Example: ft_axes = (0, 1) for 2D,
                   ft_axes = (0, 1, 2) for 3D;
                   ft_axes = None for all dimensions.
        :param batch: Batch NUFFT.
                    If provided, the shape is Nd + (batch, ).
                    The last axis is the number of parallel coils.
                    batch = None for single coil.
        :param radix: ????.
                    If provided, the shape is Nd + (batch, ).
                    The last axis is the number of parallel coils.
                    batch = None for single coil.
        :type om: numpy.float array, matrix size = (M, ndims)
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: tuple, selected axes to be transformed.
        :type batch: int or None
        :returns: 0
        :rtype: int, float
        :Example:

        >>> from pynufft import NUFFT_hsa
        >>> NufftObj = NUFFT_hsa()
        >>> NufftObj.plan(om, Nd, Kd, Jd)

        """

        self.ndims = len(Nd)  # dimension
        if ft_axes is None:
            ft_axes = range(0, self.ndims)
        self.ft_axes = ft_axes

        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes=ft_axes,
                              format='pELL', radix=radix)
        if batch is None:
            self.parallel_flag = 0
        else:
            self.parallel_flag = 1

        if batch is None:
            self.batch = numpy.uint32(1)

        else:
            self.batch = numpy.uint32(batch)

        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        #  self.sn = numpy.asarray(self.st['sn'].astype(self.dtype),
        #                            order='C')# backup
        if self.batch == 1 and (self.parallel_flag == 0):
            self.multi_Nd = self.Nd
            self.multi_Kd = self.Kd
            self.multi_M = (self.st['M'], )
            # Broadcasting the sense and scaling factor (Roll-off)
            # self.sense2 = self.sense*numpy.reshape(self.sn, self.Nd + (1, ))
        else:  # self.batch is 0:
            self.multi_Nd = self.Nd + (self.batch, )
            self.multi_Kd = self.Kd + (self.batch, )
            self.multi_M = (self.st['M'], ) + (self.batch, )
        self.invbatch = 1.0 / self.batch
        self.Kdprod = numpy.uint32(numpy.prod(self.st['Kd']))
        self.Jdprod = numpy.uint32(numpy.prod(self.st['Jd']))
        self.Ndprod = numpy.uint32(numpy.prod(self.st['Nd']))

        self.Nd_elements, self.invNd_elements = helper.strides_divide_itemsize(
                                                    self.st['Nd'])
        # only return the Kd_elements
        self.Kd_elements = helper.strides_divide_itemsize(self.st['Kd'])[0]
        self.NdCPUorder, self.KdCPUorder, self.nelem = helper.preindex_copy(
                                                        self.st['Nd'],
                                                        self.st['Kd'])
        self.offload()

        return 0

    @push_cuda_context
    def offload(self):  # API, platform_number=0, device_number=0):
        """
        self.offload():

        Off-load NUFFT to the opencl or cuda device(s)

        :param API: define the device type, which can be 'cuda' or 'ocl'
        :param platform_number: define which platform to be used.
                                The default platform_number = 0.
        :param device_number: define which device to be used.
                            The default device_number = 0.
        :type API: string
        :type platform_number: int
        :type device_number: int
        :return: self: instance
        """

        self.pELL = {}  # dictionary

        self.pELL['nRow'] = numpy.uint32(self.st['pELL'].nRow)
        self.pELL['prodJd'] = numpy.uint32(self.st['pELL'].prodJd)
        self.pELL['sumJd'] = numpy.uint32(self.st['pELL'].sumJd)
        self.pELL['dim'] = numpy.uint32(self.st['pELL'].dim)
        self.pELL['Jd'] = self.thr.to_device(
            self.st['pELL'].Jd.astype(numpy.uint32))
        self.pELL['meshindex'] = self.thr.to_device(
            self.st['pELL'].meshindex.astype(numpy.uint32))
        self.pELL['kindx'] = self.thr.to_device(
            self.st['pELL'].kindx.astype(numpy.uint32))
        self.pELL['udata'] = self.thr.to_device(
            self.st['pELL'].udata.astype(self.dtype))

        self.volume = {}

        self.volume['Nd_elements'] = self.thr.to_device(
            numpy.asarray(self.Nd_elements, dtype=numpy.uint32))
        self.volume['Kd_elements'] = self.thr.to_device(
            numpy.asarray(self.Kd_elements, dtype=numpy.uint32))
        self.volume['invNd_elements'] = self.thr.to_device(
            self.invNd_elements.astype(numpy.float32))
        self.volume['Nd'] = self.thr.to_device(numpy.asarray(
            self.st['Nd'], dtype=numpy.uint32))
        self.volume['NdGPUorder'] = self.thr.to_device(self.NdCPUorder)
        self.volume['KdGPUorder'] = self.thr.to_device(self.KdCPUorder)
        self.volume['gpu_coil_profile'] = self.thr.array(
            self.multi_Nd, dtype=self.dtype).fill(1.0)

        Nd = self.st['Nd']
        # tensor_sn = numpy.empty((numpy.sum(Nd), ), dtype=numpy.float32)
        #
        # shift = 0
        # for dimid in range(0, len(Nd)):
        #
        #     tensor_sn[shift :shift + Nd[dimid]] = \
        #     self.st['tensor_sn'][dimid][:, 0].real
        #     shift = shift + Nd[dimid]
        # self.volume['tensor_sn'] = self.thr.to_device(
        #     self.st['tensor_sn'].astype(numpy.float32))
        self.tSN = {}
        self.tSN['Td_elements'] = self.thr.to_device(
            numpy.asarray(self.st['tSN'].Td_elements, dtype=numpy.uint32))
        self.tSN['invTd_elements'] = self.thr.to_device(
            self.st['tSN'].invTd_elements.astype(numpy.float32))
        self.tSN['Td'] = self.thr.to_device(
            numpy.asarray(self.st['tSN'].Td, dtype=numpy.uint32))
        self.tSN['Tdims'] = self.st['tSN'].Tdims
        self.tSN['tensor_sn'] = self.thr.to_device(
            self.st['tSN'].tensor_sn.astype(numpy.float32))

        self.Ndprod = numpy.int32(numpy.prod(self.st['Nd']))
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.M = numpy.int32(self.st['M'])

        import reikna.fft
        if self.batch > 1:  # batch mode
            self.fft = reikna.fft.FFT(
                numpy.empty(self.st['Kd']+(self.batch, ), dtype=self.dtype),
                self.ft_axes).compile(self.thr, fast_math=False)
        else:  # elf.Reps == 1 Batch mode is wrong for
            self.fft = reikna.fft.FFT(
                numpy.empty(self.st['Kd'], dtype=self.dtype),
                self.ft_axes).compile(self.thr, fast_math=False)

        self.zero_scalar = self.dtype(0.0+0.0j)
        del self.st['pELL']
        if self.verbosity > 0:
            print('End of offload')

    @push_cuda_context
    def reset_sense(self):
        self.volume['gpu_coil_profile'].fill(1.0)

    @push_cuda_context
    def set_sense(self, coil_profile):
        if coil_profile.shape != self.multi_Nd:
            print('The shape of coil_profile is ', coil_profile.shape)
            print('But it should be', self.Nd + (self.batch, ))
            raise ValueError
        else:
            self.volume['gpu_coil_profile'] = self.thr.to_device(
                coil_profile.astype(self.dtype))
            if self.verbosity > 0:
                print('Successfully loading coil sensitivities!')

        # if coil_profile.shape == self.Nd + (self.batch, ):

    @push_cuda_context
    def to_device(self, image, shape=None):

        g_image = self.thr.array(image.shape, dtype=self.dtype)
        self.thr.to_device(image.astype(self.dtype), dest=g_image)
        return g_image

    @push_cuda_context
    def s2x(self, s):
        x = self.thr.array(self.multi_Nd, dtype=self.dtype)
#         print("Now populate the array to multi-coil")
        self.prg.cPopulate(
            self.batch,
            self.Ndprod,
            s,
            x,
            local_size=None,
            global_size=int(self.batch * self.Ndprod))
        # x2 = x  *  self.volume['gpu_coil_profile']
        # try:
        #     x2 = x  *  self.volume['gpu_coil_profile']
        # except:
        #     x2 = x
        self.prg.cMultiplyVecInplace(
            numpy.uint32(1),
            self.volume['gpu_coil_profile'],
            x,
            local_size=None,
            global_size=int(self.batch*self.Ndprod))
        # self.prg.cDistribute(
        #     self.batch,
        #     self.Ndprod,
        #     self.volume['gpu_coil_profile'],
        #     s,
        #     x,
        #     local_size=None,
        #     global_size=int(self.batch*self.Ndprod))
        return x

    @push_cuda_context
    def x2xx(self, x):
        # xx = self.thr.array(xx.shape, dtype = self.dtype)
        # self.thr.copy_array(z, dest=xx, )
        # size = int(xx.nbytes/xx.dtype.itemsize))
        # Hack of error in cuda backends; 8 is the byte of numpy.complex64
        # size = int(xx.nbytes/8)
        xx = self.thr.array(x.shape, dtype=self.dtype)
        self.thr.copy_array(x, dest=xx, )
        # size = int(xx.nbytes/xx.dtype.itemsize))
        # Hack of error in cuda backends; 8 is the byte of numpy.complex64:
        # size = int(xx.nbytes/8)

        # self.prg.cMultiplyRealInplace(
        #     self.batch,
        #     self.volume['SnGPUArray'],
        #     xx,
        #     local_size=None,
        #     global_size=int(self.Ndprod*self.batch))
        # self.prg.cTensorMultiply(numpy.uint32(self.batch),
        #                             numpy.uint32(self.ndims),
        #                             self.volume['Nd'],
        #                             self.volume['Nd_elements'],
        #                             self.volume['invNd_elements'],
        #                             self.volume['tensor_sn'],
        #                             xx,
        #                             numpy.uint32(0),
        #                             local_size=None,
        #                             global_size=int(self.batch*self.Ndprod))

        self.prg.cTensorMultiply(numpy.uint32(self.batch),
                                 numpy.uint32(self.tSN['Tdims']),
                                 self.tSN['Td'],
                                 self.tSN['Td_elements'],
                                 self.tSN['invTd_elements'],
                                 self.tSN['tensor_sn'],
                                 xx,
                                 numpy.uint32(0),
                                 local_size=None,
                                 global_size=int(self.batch*self.Ndprod))
        # self.thr.synchronize()
        return xx

    @push_cuda_context
    def xx2k(self, xx):

        """
        Private: oversampled FFT on the heterogeneous device

        First, zeroing the self.k_Kd array
        Second, copy self.x_Nd array to self.k_Kd array by cSelect
        Third, inplace FFT
        """
        k = self.thr.array(self.multi_Kd, dtype=self.dtype)
        # k = self.thr.array(self.multi_Kd, dtype=self.dtype).fill(0.0 + 0.0j)
        k.fill(0)
        # self.prg.cMultiplyScalar(self.zero_scalar,
        #                            k,
        #                            local_size=None,
        #                            global_size=int(self.Kdprod))
        # # self.prg.cSelect(self.NdGPUorder,
        #                    self.KdGPUorder,
        #                    xx,
        #                    k,
        #                    local_size=None,
        #                    global_size=int(self.Ndprod))
        # self.prg.cSelect2(self.batch,
        #                     self.volume['NdGPUorder'],
        #                     self.volume['KdGPUorder'],
        #                     xx,
        #                     k,
        #                     local_size=None,
        #                     global_size=int(self.Ndprod*self.batch))
        self.prg.cTensorCopy(
             self.batch,
             numpy.uint32(self.ndims),
             self.volume['Nd_elements'],
             self.volume['Kd_elements'],
             self.volume['invNd_elements'],
             xx,
             k,
             numpy.int32(1),  # Directions: Nd -> Kd, 1; Kd -> Nd, -1
             local_size=None,
             global_size=int(self.Ndprod))
        self.fft(k, k, inverse=False)
#         self.thr.synchronize()
        return k

    @push_cuda_context
    def k2y(self, k):
        """
        Private: interpolation by the Sparse Matrix-Vector Multiplication
        """
        # if self.parallel_flag is 1:
        #     y =self.thr.array((self.st['M'], self.batch),
        #                       dtype=self.dtype).fill(0)
        # else:
        #     y =self.thr.array( (self.st['M'], ), dtype=self.dtype).fill(0)
        y = self.thr.array(self.multi_M, dtype=self.dtype).fill(0)
        self.prg.pELL_spmv_mCoil(
                            self.batch,
                            self.pELL['nRow'],
                            self.pELL['prodJd'],
                            self.pELL['sumJd'],
                            self.pELL['dim'],
                            self.pELL['Jd'],
                            # self.pELL_currsumJd,
                            self.pELL['meshindex'],
                            self.pELL['kindx'],
                            self.pELL['udata'],
                            k,
                            y,
                            local_size=int(self.wavefront),
                            global_size=int(self.pELL['nRow'] *
                                            self.batch * self.wavefront)
                            )
#         self.thr.synchronize()
        return y

    @push_cuda_context
    def y2k(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        However, serial atomic add is far too slow and inaccurate.
        """

#         kx = self.thr.array(self.multi_Kd, dtype=numpy.float32).fill(0.0)
#         ky = self.thr.array(self.multi_Kd, dtype=numpy.float32).fill(0.0)
        k = self.thr.array(self.multi_Kd, dtype=numpy.complex64).fill(0.0)
        res = self.thr.array(self.multi_Kd, dtype=numpy.complex64).fill(0.0)
        # array which saves the residue of two sum
        
        self.prg.pELL_spmvh_mCoil(
                            self.batch,
                            self.pELL['nRow'],
                            self.pELL['prodJd'],
                            self.pELL['sumJd'],
                            self.pELL['dim'],
                            self.pELL['Jd'],
                            self.pELL['meshindex'],
                            self.pELL['kindx'],
                            self.pELL['udata'],
#                             kx, ky,
                            k,
                            res,
                            y,
                            local_size=None,
                            global_size=int(self.pELL['nRow']*self.batch )#*
#                                             int(self.pELL['prodJd']) * int(self.batch))
                            )

        return k + res

    @push_cuda_context
    def k2xx(self, k):
        """
        Private: the inverse FFT and image cropping (which is the reverse of
        _xx2k() method)
        """

        self.fft(k, k, inverse=True)
        # self.thr.synchronize()
        # self.x_Nd._zero_fill()
        # self.prg.cMultiplyScalar(self.zero_scalar,
        #                          xx,
        #                          local_size=None,
        #                          global_size=int(self.Ndprod))
        # if self.parallel_flag is 1:
            # xx = self.thr.array(self.st['Nd']+(self.batch, ),
            #                     dtype = self.dtype)
        # else:
        #     xx = self.thr.array(self.st['Nd'], dtype = self.dtype)
        xx = self.thr.array(self.multi_Nd, dtype=self.dtype)
        xx.fill(0)
        # self.prg.cSelect(self.queue,
        #                  (self.Ndprod,),
        #                  None,
        #                  self.volume['KdGPUorder'].data,
        #                  self.NdGPUorder.data,
        #                  self.k_Kd2.data,
        #                  self.x_Nd.data)
        # self.prg.cSelect2(self.batch,
        #                   self.volume['KdGPUorder'],
        #                   self.volume['NdGPUorder'],
        #                   k,
        #                   xx,
        #                   local_size=None,
        #                   global_size=int(self.Ndprod*self.batch))
        self.prg.cTensorCopy(
                             self.batch,
                             numpy.uint32(self.ndims),
                             self.volume['Nd_elements'],
                             self.volume['Kd_elements'],
                             self.volume['invNd_elements'],
                             k,
                             xx,
                             numpy.int32(-1),
                             local_size=None,
                             global_size=int(self.Ndprod))
        return xx

    @push_cuda_context
    def xx2x(self, xx):
        x = self.thr.array(xx.shape, dtype=self.dtype)
        self.thr.copy_array(xx, dest=x, )
        # size = int(xx.nbytes/xx.dtype.itemsize))
        # Hack of error in cuda backends; 8 is the byte of numpy.complex64
        # size = int(xx.nbytes/8)

        # self.prg.cMultiplyRealInplace(self.batch,
        #                               self.volume['SnGPUArray'],
        #                               z,
        #                               local_size=None,
        #                               global_size=int(self.batch*self.Ndprod))
        # self.prg.cTensorMultiply(numpy.uint32(self.batch),
        #                             numpy.uint32(self.ndims),
        #                             self.volume['Nd'],
        #                             self.volume['Nd_elements'],
        #                             self.volume['invNd_elements'],
        #                             self.volume['tensor_sn'],
        #                             x,
        #                             numpy.uint32(0),
        #                             local_size=None,
        #                             global_size=int(self.batch*self.Ndprod))
        self.prg.cTensorMultiply(numpy.uint32(self.batch),
                                 numpy.uint32(self.tSN['Tdims']),
                                 self.tSN['Td'],
                                 self.tSN['Td_elements'],
                                 self.tSN['invTd_elements'],
                                 self.tSN['tensor_sn'],
                                 x,
                                 numpy.uint32(0),
                                 local_size=None,
                                 global_size=int(self.batch *
                                                 self.Ndprod))
        # self.thr.synchronize()
        return x

    @push_cuda_context
    def x2s(self, x):
        s = self.thr.array(self.st['Nd'], dtype=self.dtype)
#         try:
        self.prg.cMultiplyConjVecInplace(
            numpy.uint32(1),
            self.volume['gpu_coil_profile'],
            x,
            local_size=None,
            global_size=int(self.batch*self.Ndprod))
#         x2 = x  *  self.volume['gpu_coil_profile'].conj()
#         except:
#             x2 = x
        self.prg.cAggregate(
            self.batch,
            self.Ndprod,
            x,
            s,
            local_size=int(self.wavefront),
            global_size=int(self.batch*self.Ndprod*self.wavefront))
        # self.prg.cMerge(self.batch,
        #                 self.Ndprod,
        #                 self.volume['gpu_coil_profile'],
        #                 x,
        #                 s,
        #                 local_size=int(self.wavefront),
        #                 global_size = int(self.batch*self.Ndprod*
        #                                   self.wavefront))
        return s

    @push_cuda_context
    def selfadjoint_one2many2one(self, gx):
        """
        selfadjoint_one2many2one NUFFT on the heterogeneous device

        :param gx: The input gpu array, with size=Nd
        :type gx: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """

        gy = self.forward_one2many(gx)
        gx2 = self.adjoint_many2one(gy)
        del gy
        return gx2

    def selfadjoint(self, gx):
        """
        selfadjoint NUFFT on the heterogeneous device

        :param gx: The input gpu array, with size=Nd
        :type gx: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """

        gy = self.forward(gx)
        gx2 = self.adjoint(gy)
        del gy
        return gx2

    @push_cuda_context
    def forward(self, gx):
        """
        Forward NUFFT on the heterogeneous device

        :param gx: The input gpu array, with size = Nd
        :type gx: reikna gpu array with dtype = numpy.complex64
        :return: gy: The output gpu array, with size = (M,)
        :rtype: reikna gpu array with dtype = numpy.complex64
        """
        try:
            xx = self.x2xx(gx)
        except:  # gx is not a gpu array
            try:
                warnings.warn('The input array may not be a GPUarray '
                              'Automatically moving the input array to gpu, '
                              'which is throttled by PCIe.', UserWarning)
                px = self.to_device(gx, )
                # pz = self.thr.to_device(numpy.asarray(gz.astype(self.dtype),
                #                                       order = 'C' ))
                xx = self.x2xx(px)
            except:
                if gx.shape != self.Nd + (self.batch, ):
                    warnings.warn('Shape of the input is ' + str(gx.shape) +
                                  ' while it should be ' +
                                  str(self.Nd+(self.batch, )), UserWarning)
                raise

        k = self.xx2k(xx)
        del xx
        gy = self.k2y(k)
        del k
        return gy

    @push_cuda_context
    def forward_one2many(self, s):
        try:
            x = self.s2x(s)
        except:  # gx is not a gpu array
            try:
                warnings.warn('In s2x(): The input array may not be '
                              'a GPUarray. Automatically moving the input'
                              ' array to gpu, which is throttled by PCIe.',
                              UserWarning)
                ps = self.to_device(s, )
                # px = self.thr.to_device(numpy.asarray(x.astype(self.dtype),
                #                                       order = 'C' ))
                x = self.s2x(ps)
            except:
                if s.shape != self.Nd:
                    warnings.warn('Shape of the input is ' + str(x.shape) +
                                  ' while it should be ' +
                                  str(self.Nd), UserWarning)
                raise

        y = self.forward(x)
        return y

    @push_cuda_context
    def adjoint_many2one(self, y):
        try:
            x = self.adjoint(y)
        except:  # gx is not a gpu array
            try:
                if self.verbosity > 0:
                    print('y.shape = ', y.shape)
                warnings.warn('In adjoint_many2one(): The input array may not '
                              'be a GPUarray. Automatically moving the input'
                              ' array to gpu, which is throttled by PCIe.',
                              UserWarning)
                py = self.to_device(y, )
                # py = self.thr.to_device(numpy.asarray(y.astype(self.dtype),
                #                                       order = 'C' ))
                x = self.adjoint(py)
            except:
                print('Failed at self.adjoint_many2one! Please check the gy'
                      ' shape, type and stride.')
                raise
        # z = self.adjoint(y)
        s = self.x2s(x)
        return s

    @push_cuda_context
    def adjoint(self, gy):
        """
        Adjoint NUFFT on the heterogeneous device

        :param gy: The input gpu array, with size=(M,)
        :type: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """
        try:
            k = self.y2k(gy)
        except:  # gx is not a gpu array
            try:
                warnings.warn('In adjoint(): The input array may not '
                              'be a GPUarray. Automatically moving the input'
                              ' array to gpu, which is throttled by PCIe.',
                              UserWarning)
                py = self.to_device(gy, )
                # py = self.thr.to_device(numpy.asarray(gy.astype(self.dtype),
                #                         order = 'C' ))
                k = self.y2k(py)
            except:
                print('Failed at self.adjont! Please check the gy shape, '
                      'type, stride.')
                raise

#             k = self.y2k(gy)
        xx = self.k2xx(k)
        del k
        gx = self.xx2x(xx)
        del xx
        return gx

    @push_cuda_context
    def release(self):
        del self.volume
        del self.prg
        del self.pELL
        self.thr.release()
        del self.thr

    @push_cuda_context
    def solve(self, gy, solver=None, *args, **kwargs):
        """
        The solver of NUFFT_hsa

        :param gy: data, reikna array, (M,) size
        :param solver: could be 'cg', 'L1TVOLS', 'L1TVLAD'
        :param maxiter: the number of iterations
        :type gy: reikna array, dtype = numpy.complex64
        :type solver: string
        :type maxiter: int
        :return: reikna array with size Nd
        """
        from ..linalg.solve_hsa import solve

        try:
            return solve(self,  gy,  solver, *args, **kwargs)
        except:
            try:
                warnings.warn('In solve(): The input array may not '
                              'be a GPUarray. Automatically moving the input'
                              ' array to gpu, which is throttled by PCIe.',
                              UserWarning)
                py = self.to_device(gy, )
                return solve(self,  py,  solver, *args, **kwargs)
            except:
                if numpy.ndarray == type(gy):
                    print("Input gy must be a reikna array with dtype ="
                          " numpy.complex64")
                    raise  # TypeError
                else:
                    print("wrong")
                    raise  # TypeError
