"""
NUFFT HSA legacy classes (deprecated)
=======================================
"""
from __future__ import absolute_import
import numpy
import scipy.sparse
import numpy.fft
# import scipy.signal
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
    Decorator: Push cude context to the top of the stack for current use
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

            
                 
class NUFFT_hsa_legacy:
    """
    (deprecated) Classical precomputed NUFFT_hsa for heterogeneous systems.
    Naive implementation of multi-dimensional NUFFT on GPU.
    Will be removed in future releases due to large memory size of 3D interpolator.
     
    .. deprecated:: 0.4. Use :class:`pynufft.NUFFT_hsa` instead.
   """
 
    def __init__(self, API = None, platform_number=None, device_number=None):
        """
        Constructor.
        :param API: The API for the heterogeneous system. API='cuda' or API='ocl'
        :param platform_number: The number of the platform found by the API. 
        :param device_number: The number of the device found on the platform. 
        :type API: string
        :type platform_number: integer 
        :type device_number: integer 
        :returns: 0
        :rtype: int, float
 
        :Example:
 
        >>> import pynufft
        >>> NufftObj = pynufft.NUFFT_hsa(API='cuda', 0, 0)        
        """
         
#         pass
        self.dtype = numpy.complex64
#         NUFFT_cpu.__init__(self)
     
        import reikna.cluda as cluda
        print('API = ', API)
        self.cuda_flag, self.ocl_flag = helper.diagnose()
        if None is API:
            if self.cuda_flag is 1:
                API = 'cuda'
            elif self.ocl_flag is 1:
                API = 'ocl'
            else:
                print('No accelerator is available.')
        else:
            api = API
        print('now using API = ', API)
        if platform_number is None:
            platform_number = 0
        if device_number is None:
            device_number = 0
         
        from reikna import cluda
        import reikna.transformations
        from reikna.cluda import functions, dtypes
        try: # try to create api/platform/device using the given parameters
            if 'cuda' == API:
                api = cluda.cuda_api()
            elif 'ocl' == API:
                api = cluda.ocl_api()
      
            platform = api.get_platforms()[platform_number]
             
            device = platform.get_devices()[device_number]
        except: # if failed, find out what's going wrong?
            print('No accelerator is detected.')
             
#             return 1
 
#         Create context from device
        self.thr = api.Thread(device) #pyopencl.create_some_context()
        print('Using opencl or cuda = ', self.thr.api)
         
#         print('Using opencl?  ', self.thr.api is reikna.cluda.ocl)
#         """
#         Wavefront: as warp in cuda. Can control the width in a workgroup
#         Wavefront is required in spmv_vector as it improves data coalescence.
#         see cCSR_spmv and zSparseMatVec
#         """
        self.wavefront = api.DeviceParameters(device).warp_size
 
        print('wavefront of OpenCL (as warp in CUDA) = ',self.wavefront)
 
 
        from ..src.re_subroutine import create_kernel_sets
        kernel_sets = create_kernel_sets(API)
                
        prg = self.thr.compile(kernel_sets, 
                                render_kwds=dict(LL =  str(self.wavefront)), 
                                fast_math=False)
        self.prg = prg        
         
        print("Note: In the future the api will change!")
        print("You have been warned!")
 
    def plan(self, om, Nd, Kd, Jd, ft_axes = None, batch = None):
        """
        Design the min-max interpolator.
         
        :param om: The M off-grid locations in the frequency domain. Normalized between [-pi, pi]
        :param Nd: The matrix size of equispaced image. Example: Nd=(256,256) for a 2D image; Nd = (128,128,128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid. Example: Kd=(512,512) for 2D image; Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size. Example: Jd=(6,6) for 2D image; Jd = (6,6,6) for a 3D image
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements. 
        :type Kd: tuple, ndims integer elements. 
        :type Jd: tuple, ndims integer elements. 
        :returns: 0
        :rtype: int, float
        :Example:
 
        >>> import pynufft
        >>> NufftObj = pynufft.NUFFT_cpu()
        >>> NufftObj.plan(om, Nd, Kd, Jd) 
         
        """         
         
 
#         n_shift = tuple(0*x for x in Nd)
        self.ndims = len(Nd) # dimension
        self.scale_gamma = numpy.prod(Kd)/numpy.prod(Nd)
        if ft_axes is None:
            ft_axes = range(0, self.ndims)
        self.ft_axes = ft_axes
#     
        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes = ft_axes, format = 'CSR')

        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        self.sn = numpy.asarray(self.st['sn'].astype(self.dtype)  ,order='C')# backup
        if batch is None:
            self.parallel_flag = 0
        else:
            self.parallel_flag = 1
             
        if batch is None:
            self.batch = numpy.uint32(1)
#             self.nobatch = 1
        else:
            self.batch = numpy.uint32(batch)
  
        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        self.sn = numpy.asarray(self.st['sn'].astype(self.dtype)  ,order='C')# backup
         
        if self.batch == 1 and (self.parallel_flag == 0):
            self.multi_Nd =   self.Nd
            self.multi_Kd =   self.Kd
            self.multi_M =   (self.st['M'], )      
#             self.sense2 = self.sense*numpy.reshape(self.sn, self.Nd + (1, )) # broadcasting the sense and scaling factor (Roll-off)
        else: #self.batch is 0:
            self.multi_Nd =   self.Nd + (self.batch, )
            self.multi_Kd =   self.Kd + (self.batch, )
            self.multi_M =   (self.st['M'], )+ (self.batch, )         
        # Calculate the density compensation function
        self.sp = self.st['p'].copy().tocsr()
        self.spH = (self.st['p'].getH().copy()).tocsr()        
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.Jdprod = numpy.int32(numpy.prod(self.st['Jd']))
        del self.st['p'], self.st['sn']
#         self._precompute_sp()        
#         del self.st['p0'] 
        self.NdCPUorder, self.KdCPUorder, self.nelem =     helper.preindex_copy(self.st['Nd'], self.st['Kd'])
#         self.Nd_elements, self.Kd_elements, self.invNd_elements = helper.preindex_copy2(self.st['Nd'], self.st['Kd'])
        self.Nd_elements,  self.invNd_elements = helper.strides_divide_itemsize(self.st['Nd'])
        self.Kd_elements = helper.strides_divide_itemsize( self.st['Kd'])[0] # only return the Kd_elements
        self.offload()
         
        return 0
     
    @push_cuda_context
    def offload(self):
        """
        self.offload():
         
        Off-load NUFFT to the opencl or cuda device(s)
         
        :param API: define the device type, which can be 'cuda' or 'ocl'
        :param platform_number: define which platform to be used. The default platform_number = 0.
        :param device_number: define which device to be used. The default device_number = 0.
        :type API: string
        :type platform_number: int
        :type device_number: int
        :return: self: instance
 
        """

        self.volume = {}
        self.volume['NdGPUorder'] =  self.thr.to_device( self.NdCPUorder)
        self.volume['KdGPUorder'] =  self.thr.to_device( self.KdCPUorder)
        self.volume['SnGPUArray'] = self.thr.to_device(  self.sn)
#         self.NdGPUorder = self.thr.to_device( self.NdCPUorder)
#         self.KdGPUorder =  self.thr.to_device( self.KdCPUorder)
        self.Ndprod = numpy.int32(numpy.prod(self.st['Nd']))
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.M = numpy.int32( self.st['M'])
         
#         self.SnGPUArray = self.thr.to_device(  self.sn)
        self.csr = {}
        self.csrH = {}
        self.csr['data'] = self.thr.to_device( self.sp.data.astype(self.dtype))
        self.csr['indices'] = self.thr.to_device( self.sp.indices.astype(numpy.uint32))
        self.csr['indptr'] =  self.thr.to_device( self.sp.indptr.astype(numpy.uint32))
        self.csr['numrow'] = self.M
        self.csr['numcol'] = self.Kdprod
         
 
        del self.sp
        self.csrH['data'] = self.thr.to_device(  self.spH.data.astype(self.dtype))
        self.csrH['indices'] =  self.thr.to_device(  self.spH.indices.astype(numpy.uint32))
        self.csrH['indptr'] =  self.thr.to_device(  self.spH.indptr.astype(numpy.uint32))
        self.csrH['numrow'] = self.Kdprod
         
        del self.spH
 
        import reikna.fft
 
        self.fft = reikna.fft.FFT(numpy.empty(self.st['Kd'], dtype=self.dtype), self.ft_axes).compile(self.thr, fast_math=False)
 
        self.zero_scalar=self.dtype(0.0+0.0j)
         
    @push_cuda_context
    def release(self):
        del self.volume
        del self.csr
        del self.csrH
        del self.prg
        self.thr.release()
        del self.thr
         
    @push_cuda_context
    def solve(self,gy, solver=None, *args, **kwargs):
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
        self._precompute_sp()            
        try:
            return solve(self,  gy,  solver, *args, **kwargs)
        except:
            try:
                    print('The input array may not be a GPUarray.')
                    print('Automatically moving the input array to gpu, which is throttled by PCIe.')
                    print('You have been warned!')
                    py = self.thr.to_device(numpy.asarray(gy.astype(self.dtype),  order = 'C' ))
                    return solve(self,  py,  solver, *args, **kwargs)
            except:
                if numpy.ndarray==type(gy):
                    print("input gy must be a reikna array with dtype = numpy.complex64")
                    raise #TypeError
                else:
                    print("wrong")
                    raise #TypeError
                 
    @push_cuda_context
    def forward(self, gx):
            """
            Forward NUFFT on the heterogeneous device
             
            :param gx: The input gpu array, with the size of Nd or Nd + (batch, )
            :type: reikna gpu array with the dtype of numpy.complex64
            :return: gy: The output gpu array, with size of (M,) or (M, batch)
            :rtype: reikna gpu array with the dtype of numpy.complex64
            """
             
            try:
                xx = self.x2xx(gx)
            except: # gx is not a gpu array 
                try:
                    print('The input array may not be a GPUarray.')
                    print('Automatically moving the input array to gpu, which is throttled by PCIe.')
                    print('You have been warned!')
                    px = self.thr.to_device(numpy.asarray(gx.astype(self.dtype),  order = 'C' ))
                    xx = self.x2xx(px)
                except:
                    if gx.shape != self.Nd:
                        print('shape of the input = ', gx.shape, ', but it should be ', self.Nd)
                    raise
                 
            k = self.xx2k(xx)
            del xx
            gy = self.k2y(k)
            del k
            return gy
         
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
            except: # gx is not a gpu array 
                try:
                    print('The input array may not be a GPUarray.')
                    print('Automatically moving the input array to gpu, which is throttled by PCIe.')
                    print('You have been warned!')
                    py = self.thr.to_device(numpy.asarray(gy.astype(self.dtype),  order = 'C' ))
                    k = self.y2k(py)
                except:
                    print('Failed at self.adjont! Please check the gy shape, type, stride.')
                    raise
                             
#             k = self.y2k(gy)
            xx = self.k2xx(k)
            del k
            gx = self.xx2x(xx)
            del xx
            return gx
         
    @push_cuda_context
    def selfadjoint2(self, gx):
        """
        The Toeplitz on the heterogeneous device using the diaognal matrix to compute the convolution. 
        It is an approximation to the fully selfadjoint but in some cases the accuracy is high enough.
         
        :param gx: The input gpu array, with size=Nd
        :type: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """                
        gk = self.xx2k(gx)
#         try:
        self.prg.cMultiplyVecInplace(self.batch, self.W, gk, local_size=None, global_size=int(self.Kdprod))
#         except:
#             self._precompute_sp()
#             self.prg.cMultiplyVecInplace(self.W, gk, local_size=None, global_size=int(self.Kdprod))
             
        gx2 = self.k2xx(gk)
        del gk
         
         
        return gx2
 
    @push_cuda_context
    def selfadjoint(self, gx):
        """
        selfadjoint.
        
        :param gx: The input gpu array, with size=Nd
        :type: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """                
        gy = self.forward(gx)
        gx2 = self.adjoint(gy)
        del gy
        return gx2    
     
    @push_cuda_context
    def x2z(self, x):
        z = self.thr.array(self.multi_Nd, dtype=self.dtype)
        self.thr.copy_array(x, z, )#size = int(x.nbytes/x.dtype.itemsize))#src_offset, dest_offset, size)        
        return z
     
    @push_cuda_context
    def z2xx(self, x):
        """
        Private: Scaling on the heterogeneous device
        Inplace multiplication of self.x_Nd by the scaling factor self.SnGPUArray.
        """           
        xx = self.thr.array(self.multi_Nd, dtype=self.dtype)
        self.thr.copy_array(x, xx, )#size = int(x.nbytes/x.dtype.itemsize))#src_offset, dest_offset, size)
        self.prg.cMultiplyVecInplace(self.batch, self.volume['SnGPUArray'], xx, local_size=None, global_size=int(self.Ndprod))
#         self.thr.synchronize()
        return xx
     
    @push_cuda_context
    def xx2k(self, xx):
        """
        Private: oversampled FFT on the heterogeneous device
         
        Firstly, zeroing the self.k_Kd array
        Second, copy self.x_Nd array to self.k_Kd array by cSelect
        Third: inplace FFT
        """
        k = self.thr.array(self.multi_Kd, dtype = self.dtype)
        k.fill(0)
#         self.prg.cMultiplyScalar(self.zero_scalar, k, local_size=None, global_size=int(self.Kdprod))
        self.prg.cSelect(self.volume['NdGPUorder'],      self.volume['KdGPUorder'],  xx, k, local_size=None, global_size=int(self.Ndprod))
        self.fft( k, k,inverse=False)
        self.thr.synchronize()
        return k
     
    @push_cuda_context  
    def k2y(self, k):
        """
        Private: interpolation by the Sparse Matrix-Vector Multiplication
        """
        y =self.thr.array( self.multi_M, dtype=self.dtype).fill(0)
        self.prg.cCSR_spmv_vector(                                
                           self.csr['numrow'], 
                           self.csr['indptr'],
                           self.csr['indices'],
                           self.csr['data'], 
                           k,
                           y,
                           local_size=int(self.wavefront),
                           global_size=int(self.csr['numrow']*self.wavefront) 
                            )
 
        self.thr.synchronize()
        return y    
    @push_cuda_context
    def y2k(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        """
        k = self.thr.array(self.multi_Kd, dtype = self.dtype)
    
        self.prg.cCSR_spmv_vector(
                           self.csrH['numrow'], 
                           self.csrH['indptr'],
                           self.csrH['indices'],
                           self.csrH['data'], 
                           y,
                           k,
                           local_size=int(self.wavefront),
                           global_size=int(self.csrH['numrow']*self.wavefront) 
                            )#,g_times_l=int(csrnumrow))
 
        self.thr.synchronize()
        return k    
    @push_cuda_context
    def k2xx(self, k):
        """
        Private: the inverse FFT and image cropping (which is the reverse of _xx2k() method)
        """        
        xx = self.thr.array(self.multi_Nd, dtype = self.dtype)
        self.fft( k, k, inverse=True)
        self.thr.synchronize()
#         self.x_Nd._zero_fill()
#         self.prg.cMultiplyScalar(self.zero_scalar, xx,  local_size=None, global_size=int(self.Ndprod ))
        xx.fill(0)
#         self.prg.cSelect(self.queue, (self.Ndprod,), None,   self.KdGPUorder.data,  self.NdGPUorder.data,     self.k_Kd2.data, self.x_Nd.data )
        self.prg.cSelect(  self.volume['KdGPUorder'],  self.volume['NdGPUorder'],     k, xx, local_size=None, global_size=int(self.Ndprod ))
         
        return xx
    @push_cuda_context
    def x2xx(self, x):
        z = self.x2z(x)
        xx = self.z2xx(z)
        del z
        return xx
     
    @push_cuda_context
    def xx2x(self, xx):
        """
        Private: rescaling, which is identical to the  _x2xx() method
        """
        z = self.xx2z(xx)
        x = self.z2x(z)
        del z
        return x
     
    @push_cuda_context
    def xx2z(self, xx):
        z = self.thr.array(self.multi_Nd, dtype=self.dtype)
        self.thr.copy_array(xx, z, )#size = int(x.nbytes/x.dtype.itemsize))#src_offset, dest_offset, size)
        self.prg.cMultiplyConjVecInplace(self.batch, self.volume['SnGPUArray'], z, local_size=None, global_size=int(self.Ndprod))
        self.thr.synchronize()        
        return z
    def z2x(self, z):
        return self.x2z(z)
     
    @push_cuda_context
    def _precompute_sp(self):    
        y =self.thr.array( self.multi_M, dtype=self.dtype).fill(1.0)
        M = self.adjoint(y)
        W = self.xx2k(M)
        del M
        self.W = self.thr.array( self.multi_Kd, dtype=self.dtype)
        self.thr.copy_array(W, self.W, )
        self.prg.cMultiplyConjVecInplace(numpy.uint32(1), W, self.W, local_size=None, global_size =  int(self.batch * self.Kdprod))
        del W
        self.prg.cSqrt(self.W, local_size=None, global_size =  int( self.Kdprod))
         
    @push_cuda_context
    def to_device(self, image, shape = None):
         
        g_image = self.thr.array(image.shape, dtype = self.dtype)
        self.thr.to_device(image.astype(self.dtype), dest = g_image)
        return g_image        
        

                   
