"""
HSA solvers
======================================
"""

"""
- Bugfix: fix the instability of cg due to alpha and beta which must be fetched from the device
"""
import numpy, scipy
dtype = numpy.complex64
# def  L1TVLAD():
import scipy
import numpy
from ..src._helper import helper

def cDiff(x, d_indx):
        """
        (stable) Compute image gradient
        Work with indxmap_diff(Nd).
        ...
        """    
        a2=numpy.asarray(x.copy(),order='C')
        a2.flat =   a2 .flat[d_indx] - a2 .flat
        return a2
def _create_kspace_sampling_density(nufft):
        """
        (stable) Compute k-space sampling density from the nufft object
        """    
        y = numpy.ones((nufft.st['M'],),dtype = numpy.complex64)
        gy = nufft.thr.to_device(y)
        gk = nufft._y2k_device(gy)
        w =  numpy.abs(gk.get())#**2) ))
    
        nufft.st['w'] = w#self.nufftobj.vec2k(w)
        RTR=nufft.st['w'] # see __init__() in class "nufft"
        return RTR
# def _create_laplacian_kernel(nufft):
# #===============================================================================
# # #        # Laplacian oeprator, convolution kernel in spatial domain
# #         # related to constraint
# #===============================================================================
#     uker = numpy.zeros(nufft.st['Kd'][:],dtype=numpy.complex64,order='C')
#     n_dims= numpy.size(nufft.st['Nd'])
# 
#     if n_dims == 1:
#         uker[0] = -2.0
#         uker[1] = 1.0
#         uker[-1] = 1.0
#     elif n_dims == 2:
#         uker[0,0] = -4.0
#         uker[1,0] = 1.0
#         uker[-1,0] = 1.0
#         uker[0,1] = 1.0
#         uker[0,-1] = 1.0
#     elif n_dims == 3:  
#         uker[0,0,0] = -6.0
#         uker[1,0,0] = 1.0
#         uker[-1,0,0] = 1.0
#         uker[0,1,0] = 1.0
#         uker[0,-1,0] = 1.0
#         uker[0,0,1] = 1.0
#         uker[0,0,-1] = 1.0                      
# 
#     uker =numpy.fft.fftn(uker) #, self.nufftobj.st['Kd'], range(0,numpy.ndim(uker)))
#     return uker  
# def GBPDNA(nufft. gy, maxiter, rho):
#     def A(x):
#         gy = nufft.forward(x)
#         return gy
#     def AH(gy):
#         x2 = nufft.adjoint(gy)
#         return x2
    
def L1TVOLS(nufft, gy, maxiter, rho  ): # main function of solver
    """
    L1-total variation regularized ordinary least square 
    """
    mu = 1.0
    LMBD = rho*mu

    def AHA(x):
        x2 = nufft._selfadjoint_device(x)
        return x2
    def AH(gy):
        x2 = nufft._adjoint_device(gy)
        return x2
    
    uker_cpu = mu*_create_kspace_sampling_density(nufft)   - LMBD* helper.create_laplacian_kernel(nufft) # on cpu
    uker = nufft.thr.to_device(uker_cpu.astype(numpy.complex64))
    AHy = AH(gy) # on  device?
    z = numpy.zeros(nufft.st['Nd'],dtype = numpy.complex64,order='C')
    z_gpu = nufft.thr.to_device(z)
    xkp1 = nufft.thr.copy_array(z_gpu)
    AHyk = nufft.thr.copy_array(z_gpu)
           
#         self._allo_split_variables()        
    zz= []
    bb = []
    dd = []
    d_indx, dt_indx = helper.indxmap_diff(nufft.st['Nd'])
    ndims = len(nufft.st['Nd'])
    s_tmp = []
    for pp in range(0, ndims):
        s_tmp += [0, ]
    for jj in range(    0,  ndims): # n_dims + 1 for wavelets
        d_indx[jj] = nufft.thr.to_device(d_indx[jj])
        dt_indx[jj] = nufft.thr.to_device(dt_indx[jj])
#     z=numpy.zeros(nufft.st['Nd'], dtype = nufft.dtype, order='C')
    
#     ndims = len(nufft.st['Nd'])
    
    for jj in range(    0,  ndims): # n_dims + 1 for wavelets
        
        zz += [nufft.thr.copy_array(z_gpu),]
        bb += [nufft.thr.copy_array(z_gpu),]
        dd +=  [nufft.thr.copy_array(z_gpu),]
#     zf = nufft.thr.copy_array(z_gpu)
#     bf = nufft.thr.copy_array(z_gpu)
#     df = nufft.thr.copy_array(z_gpu)

    n_dims = len(nufft.st['Nd'])#numpy.size(uf.shape)
    
    tmp_gpu = nufft.thr.copy_array(z_gpu) 
    
    
    for outer in numpy.arange(0, maxiter):
#             for inner in numpy.arange(0,nInner):
            
        # solve Ku = rhs
#                 rhs = (mu*(AHyk + df - bf) +  # right hand side
#                 LMBD*(cDiff(dd[0] - bb[0],  dt_indx[0])) + 
#                 LMBD*(cDiff(dd[1] - bb[1],  dt_indx[1]))  )      
        rhs = nufft.thr.copy_array(AHyk)
        
#         rhs += df
#         
#         rhs -= bf
        
#         rhs *= mu
        nufft.prg.cMultiplyScalar( dtype(mu), rhs, local_size=None, global_size=int(nufft.Ndprod))
        
#         print(rhs.get())
        for pp in range(0, ndims): 
            in_cDiff = nufft.thr.copy_array(dd[pp])
            
            in_cDiff -= bb[pp]
            
    #         out_cDiff = nufft.thr.empty_like(in_cDiff) 
            nufft.prg.cDiff(dt_indx[pp], in_cDiff, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
            
    #         tmp_gpu *= LMBD
            nufft.prg.cMultiplyScalar( dtype(LMBD), tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
            
            rhs += tmp_gpu
        
#         print(rhs.get())
#         in_cDiff = nufft.thr.copy_array(dd[1])
#         
#         in_cDiff -= bb[1]
#         
# #         out_cDiff = nufft.thr.empty_like(in_cDiff) 
#         nufft.prg.cDiff(dt_indx[1], in_cDiff, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         
# #         tmp_gpu *= LMBD
#         nufft.prg.cMultiplyScalar( dtype(LMBD), tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         
#         rhs += tmp_gpu
        
#         print(rhs.get())
    
        # Note K = F' uker F
        # so K-1 ~ F
#         xkp1 = nufft.k2xx(nufft.xx2k(rhs) / uker) 
        xx = nufft.thr.copy_array(rhs)
        
        k = nufft._xx2k_device(xx)
        
        k /= uker
        
#         nufft.k_Kd2 = nufft.thr.copy_array(nufft.k_Kd)
        xkp1 = nufft._k2xx_device(k)
#         xkp1 = nufft.thr.copy_array(nufft.x_Nd)
#         print(xkp1.get())
#                 self._update_d(xkp1)


#         zz[0] = cDiff(xkp1,  d_indx[0])
#         zz[1] = cDiff(xkp1,  d_indx[1])
        for pp in range(0, ndims):
            nufft.prg.cDiff(d_indx[pp],  xkp1, zz[pp],  local_size=None, global_size=int(nufft.Ndprod)) 
#         nufft.prg.cDiff(d_indx[0],  xkp1, zz[0],  local_size=None, global_size=int(nufft.Ndprod))
#         nufft.prg.cDiff(d_indx[1],  xkp1, zz[1],  local_size=None, global_size=int(nufft.Ndprod))
        

#         zf = AHA(xkp1)  -AHy
        zf = AHA(xkp1)
        
        zf -= AHy 
        

        '''
        soft-thresholding the edges
        '''
        for pp in range(0, ndims):
            s_tmp[pp] = zz[pp] + bb[pp]
#         s1 = zz[0] + bb[0]
#         
#         s2 = zz[1] + bb[1]
        
#         s = s1**2 + s2**2
#         s1 *= s1
#             nufft.prg.cMultiplyConjVec(s_tmp[pp], s_tmp[pp], tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
        for pp in range(0, ndims):
            if pp > 0:
#                 s += tmp_gpu
                nufft.prg.cHypot(s, s_tmp[pp], local_size=None, global_size=int(nufft.Ndprod))
#                 nufft.thr.synchronize()
            else: # pp == 0
                s = nufft.thr.copy_array(s_tmp[pp])
#         nufft.prg.cMultiplyConjVec(s1, s1, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         s = nufft.thr.copy_array(tmp_gpu)
# #         s2 *= s2
#         nufft.prg.cMultiplyConjVec(s2, s2, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         s += tmp_gpu
        
#         s = s1 + s2
        
        
#         nufft.prg.cSqrt(s, local_size=None, global_size=int(nufft.Ndprod))
        
        s += 1e-6
        
        threshold_value = dtype(1/LMBD)
#         r =(s > threshold_value)*(s-threshold_value)/s#numpy.maximum(s - threshold_value ,  0.0)/s
        nufft.prg.cAnisoShrink(threshold_value, s, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
        tmp_gpu /=s
        
#         dd[0] = s1*r
#         dd[1] = s2*r
#         dd[0] = s1*tmp_gpu
        for pp in range(0, ndims):
            nufft.prg.cMultiplyVec(s_tmp[pp], tmp_gpu, dd[pp], local_size=None, global_size=int(nufft.Ndprod)) 
#         nufft.prg.cMultiplyVec(s1, tmp_gpu, dd[0], local_size=None, global_size=int(nufft.Ndprod)) 
        
#         dd[1] = s2*tmp_gpu
        
#         nufft.prg.cMultiplyVec(s2, tmp_gpu, dd[1], local_size=None, global_size=int(nufft.Ndprod))
        
#         tmp_gpu = zf+bf
        
#         threshold_value=dtype(1.0/mu)
        
#         df.real =0.0+ (df.real>threshold_value)*(df.real - threshold_value) +(df.real<= - threshold_value)*(df.real+threshold_value)
#         df.imag = 0.0+(df.imag>threshold_value)*(df.imag - threshold_value) +(df.imag<= - threshold_value)*(df.imag+threshold_value)
    
#         nufft.prg.cAnisoShrink(threshold_value,  tmp_gpu, df, local_size=None, global_size=int(nufft.Ndprod))
        
         
#                 df =     sy
        # end of shrinkage
        for pp in range(0, ndims):
            bb[pp] += zz[pp] - dd[pp]
#         bb[0] += zz[0] - dd[0] 
#         bb[1] += zz[1] - dd[1] 
#         bf += zf - df 
#                 self._update_b() # update b based on the current u
#         print(outer)
        
        AHyk -= zf # Linearized Bregman iteration f^k+1 = f^k + f - Au
        
#     print(xkp1.get())
#             print(outer)
#     print('here')
#     nufft.x_Nd = nufft.thr.copy_array(xkp1)
    return xkp1

def _pipe_density(nufft,maxiter):
    """
    Private: create the density function in the data space by a iterative solution
    Pipe et al. 1999
    """


    try:
        if maxiter < nufft.last_iter:
         
            W = nufft.st['W']
        else: #maxiter > nufft.last_iter
            W = nufft.st['W']
            for pp in range(0,maxiter - nufft.last_iter):

#             E = nufft.st['p'].dot(V1.dot(W))
                E = nufft.forward(nufft.adjoint(W))
                W = (W/E)             
            nufft.last_iter = maxiter   
    except:
        W = nufft.thr.copy_array(nufft.y)
#         nufft.prg.cMultiplyScalar(nufft.zero_scalar, W, local_size=None, global_size=int(nufft.M))
        W.fill(0.0 + 0.0j)
#         V1= nufft.st['p'].getH()
    #     VVH = V.dot(V.getH()) 
         
        for pp in range(0,1):
#             E = nufft.st['p'].dot(V1.dot(W))

            E = nufft._forward_device(nufft._adjoint_device(W))
            W /= E
#                 nufft.prg.cMultiplyVecInplace(self.SnGPUArray, self.x_Nd, local_size=None, global_size=int(self.Ndprod))
    
    return W  

def solve(nufft,gy, solver=None,  maxiter=30, *args, **kwargs):
    """
    The solve function of NUFFT_hsa.
    The current version supports solvers = 'cg' or 'L1TVOLS'. 
    
    :param nufft: NUFFT_hsa object
    :param y: (M,) or (M, batch) array, non-uniform data. If batch is provided, 'cg' and 'L1TVOLS' returns different image shape.
    :type y: numpy.complex64 reikna array
    :return: x: Nd or Nd + (batch, ) image. L1TVOLS always returns Nd. 'cg' returns Nd + (batch, ) in batch mode. 
    :rtype: x: reikna array, complex64. 
    """
    # define the reduction kernel on the device
#     if None ==  solver:
#         solver  =   'cg'
    if 'L1TVLAD' == solver:
        x2=L1TVLAD(nufft, gy, maxiter=maxiter, *args, **kwargs  )
#         x2 = nufft.thr.copy_array(nufft.x_Nd)
        return x2
    elif 'L1TVOLS' == solver:
        x2=L1TVOLS(nufft, gy, maxiter=maxiter, *args, **kwargs  )
#         x2 = nufft.thr.copy_array(nufft.x_Nd)
        return x2
    elif 'dc'   ==  solver:
         """
         Density compensation method
         nufft.st['W'] will be computed if doesn't exist
         If nufft.st['W'] exist then x2 = nufft.adjoint(nufft.st['W']*y)
         input: 
             y: (M,) array
         output:
             x2: Nd array
         """
         print(solver, ":density compensation method. I won't recommend it as the GPU version is not needed! Try the CPU version")

#          nufft.st['W'] = nufft._pipe_density(maxiter=maxiter,*args, **kwargs)
#  
#          x2 = nufft.adjoint(nufft.st['W']*gy)
         return x2
#             return gx        
    elif 'cg' == solver:
 
        from reikna.algorithms import Reduce, Predicate, predicate_sum
         
        nufft.reduce_sum = Reduce(numpy.zeros(nufft.Kd, dtype = nufft.dtype), predicate_sum(dtype)).compile(nufft.thr)      
#         nufft.reduce_sum  = nufft.reduce_sum.compile(nufft.thr)        
         
        
        # update: b = spH * gy         
        b = nufft._y2k_device(gy)
        
        # Initialize x = b
        x   =   nufft.thr.copy_array( b)
        rsold = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
#         rsold.fill(0.0+0.0j)
        nufft.reduce_sum(rsold, x)
#         print('x',rsold) 
         
        # initialize r = b - A * x
        r   =   nufft.thr.empty_like( b)
        
#         r.fill(0.0 + 0.0j) 
        y_tmp = nufft._k2y_device(x)
        
        Ax = nufft._y2k_device(y_tmp)
        
        del y_tmp
        rsold = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
#         rsold.fill(0.0 + 0.0j)
        nufft.reduce_sum(rsold, Ax)
#         print('Ax',rsold) 
        nufft.prg.cAddVec(b, - Ax, r , local_size=None, global_size = int(nufft.batch * nufft.Kdprod))
        
#         nufft.thr.synchronize()
        # p = r
        p   =   nufft.thr.copy_array(r)
        
        # rsold = r' * r
        tmp_array = nufft.thr.empty_like( r)
#         tmp_array.fill(0.0 + 0.0j)
        nufft.prg.cMultiplyConjVec(r, r, tmp_array, local_size=None, global_size=int(nufft.batch * nufft.Kdprod))
#         nufft.thr.synchronize()
        rsold = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
#         rsold.fill(0.0 + 0.0j)
        nufft.reduce_sum(rsold, tmp_array)
 
        # allocate Ap
#         Ap  =   nufft.thr.empty_like( b)     
 
        rsnew = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
#         rsnew.fill(0.0 + 0.0j)
        tmp_sum = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
#         tmp_sum.fill(0.0 + 0.0j)
        for pp in range(0, maxiter):
            
            tmp_p = nufft._k2y_device(p)
            Ap = nufft._y2k_device(tmp_p)
            del tmp_p
#             alpha = rs_old/(p'*Ap)
            nufft.prg.cMultiplyConjVec(p, Ap, tmp_array, local_size=None, global_size=int(nufft.batch * nufft.Kdprod))
#             nufft.thr.synchronize()
            nufft.reduce_sum(tmp_sum, tmp_array)
            
            alpha = rsold / tmp_sum
#             alpha_cpu = alpha.get()
#             if numpy.isnan(alpha_cpu):
#                 alpha_cpu = 0 # avoid singularity
                
#             print(tmp_sum, alpha, rsold)
#             print(pp,rsold , alpha, numpy.sum(tmp_array.get()) )
            # x = x + alpha*p
            p2 = nufft.thr.copy_array(p)
            
            nufft.prg.cMultiplyScalar(alpha.get(), p2,  local_size=None, global_size=int(nufft.batch * nufft.Kdprod))
#             nufft.thr.synchronize()
#             nufft.prg.cAddVec(x, alpha, local_size=None, global_size=int(nufft.Kdprod))
            x += p2
 
            # r = r - alpha * Ap
            p2= nufft.thr.copy_array(Ap)
#             nufft.thr.synchronize()
            nufft.prg.cMultiplyScalar(alpha.get(), p2,  local_size=None, global_size=int(nufft.batch * nufft.Kdprod))
#             nufft.thr.synchronize()
            r -= p2
#             print(pp, numpy.sum(x.get()), numpy.sum(r.get()))
            # rs_new = r'*r
             
            nufft.prg.cMultiplyConjVec(r,    r,  tmp_array, local_size=None, global_size=int(nufft.batch * nufft.Kdprod))
#             nufft.thr.synchronize()
            nufft.reduce_sum(rsnew, tmp_array)        
             
            # tmp_sum = p = r + (rs_new/rs_old)*p
            beta = rsnew/rsold
#             beta_cpu = beta.get()
#             if numpy.isnan(beta_cpu):
#                 beta_cpu = 0
#             print(beta, rsnew, rsold)
            p2= nufft.thr.copy_array(p)
            nufft.prg.cMultiplyScalar(beta.get(),   p2,  local_size=None, global_size=int(nufft.batch * nufft.Kdprod))
#             nufft.thr.synchronize()
            nufft.prg.cAddVec(r, p2, p, local_size=None, global_size=int(nufft.batch * nufft.Kdprod))
#             nufft.thr.synchronize()
            p = r + p2
             
            rsold =nufft.thr.copy_array( rsnew)
#             nufft.thr.synchronize()
        # end of iteration    
         
        # copy result to k_Kd2
#         nufft.k_Kd2 = nufft.thr.copy_array(x)
         
        # inverse FFT: k_Kd2 -> x_Nd
        x2 = nufft._k2xx_device(x) # x is the solved k space
        
        # rescale the SnGPUArray
        # x2 /= nufft.volume['gpu_sense2']
#         x3 = nufft.x2s(x2) # combine multi-coil to single-coil
        try:
            x2 /= nufft.volume['SnGPUArray']
        except:
            
            nufft.prg.cTensorMultiply(numpy.uint32(nufft.batch), 
                                    numpy.uint32(nufft.tSN['Tdims']),
                                    nufft.tSN['Td'],
                                    nufft.tSN['Td_elements'],
                                    nufft.tSN['invTd_elements'],
                                    nufft.tSN['tensor_sn'], 
                                    x2, 
                                    numpy.uint32(1), # division, 1 is true
                                    local_size = None, global_size = int(nufft.batch*nufft.Ndprod))
         
        return x2