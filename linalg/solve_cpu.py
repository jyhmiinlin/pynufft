"""
CPU solvers
======================================
"""

import scipy
import scipy.sparse.linalg
import numpy
from ..src._helper import helper

def cDiff(x, d_indx):
        """
        Compute image gradient, which needs the results of indxmap_diff(Nd)
        :param x: The image array
        :param d_indx: The index of the shifted image
        :type x: numpy.float array, matrix size = Nd
        :type d_indx: int32
        :returns: x_diff: Image gradient determined by d_indx
        :rtype: x_diff: numpy.complex64      
        """
        x_diff=numpy.asarray(x.copy(),order='C')
        x_diff.flat =   x_diff.ravel()[d_indx] - x.ravel()
        return x_diff
    
def _create_kspace_sampling_density(nufft):
        """
        Compute k-space sampling density
        """    
        y = numpy.ones(nufft.st['M'],dtype = numpy.complex64)
#         w = numpy.abs(nufft.xx2k(nufft.adjoint(y)))
        
        if nufft.parallel_flag is 1:
            w =  numpy.abs( nufft.xx2k(nufft.adjoint(y)))[..., 0]#**2) ))
        else:
            w =  numpy.abs( nufft.xx2k(nufft.adjoint(y)))#**2) ))
        nufft.st['w'] = w#self.nufftobj.vec2k(w)
        RTR=nufft.st['w'] # see __init__() in class "nufft"
#         print('RTR.shape = ', RTR.shape)
        return RTR
   
    
def L1TVOLS(nufft, y, maxiter, rho ): # main function of solver
    """
    L1-total variation regularized ordinary least square 
    """
    mu = 1.0
    LMBD = rho*mu

    def AHA(x):
        x2 = nufft.selfadjoint(x)
        return x2
    def AH(y):
        
        x2 = nufft.adjoint(y.reshape(nufft.st['M'], order='C'))
        return x2
    
        
    
    uker = mu*_create_kspace_sampling_density(nufft)
#     print('uker.shape', uker.shape) 
    uker = uker - LMBD* helper.create_laplacian_kernel(nufft)
#     print('uker.shape', uker.shape)
#     import matplotlib.pyplot
#     matplotlib.pyplot.imshow(abs(uker))
#     matplotlib.pyplot.show()
#     print('y.shape=', y.shape)
    AHy = AH(y)
#     print('AHy.shape = ', AHy.shape)
    
    xkp1 = numpy.zeros_like(AHy)
    AHyk = numpy.zeros_like(AHy)
           
#         self._allo_split_variables()        
    zz= []
    bb = []
    dd = []
    d_indx, dt_indx = helper.indxmap_diff(nufft.st['Nd'])
    z=numpy.zeros(nufft.st['Nd'], dtype = nufft.dtype, order='C')
    
    ndims = len(nufft.st['Nd'])
    s_tmp = []
    for pp in range(0,ndims):
            s_tmp += [0, ]
            
            
    for jj in range(    0,  ndims): # n_dims + 1 for wavelets
        
        zz += [z.copy(),]
        bb += [z.copy(),]
        dd +=  [z.copy(),]
#     zf = z.copy()
#     bf = z.copy()
#     df = z.copy()

    n_dims = len(nufft.st['Nd'])#numpy.size(uf.shape)
    
    for outer in numpy.arange(0, maxiter):
#             for inner in numpy.arange(0,nInner):
            
        # solve Ku = rhs
            
        rhs = mu*AHyk# + df - bf)   # right hand side
        for pp in range(0,ndims):
#             diff1 =  dd[pp] - bb[pp]
#             diff1 = numpy.roll( diff1, 1, axis = pp) - diff1
            rhs += LMBD*(cDiff(dd[pp] - bb[pp], dt_indx[pp]))
#             del diff1
#                 LMBD*(cDiff(dd[1] - bb[1],  dt_indx[1]))  )          
        # Note K = F' uker F
        # so K-1 ~ F
        xkp1 = nufft.k2xx_one2one( (nufft.xx2k_one2one(rhs)+1e-7) / (uker+1e-7)) 
#                 self._update_d(xkp1)
        for pp in range(0,ndims):
            
#             zz[pp] = numpy.roll( xkp1, -1, axis = pp) - xkp1            
            zz[pp] = cDiff(xkp1, d_indx[pp])
#         zz[0] = cDiff(xkp1,  d_indx[0])
#         zz[1] = cDiff(xkp1,  d_indx[1])
        zf = AHA(xkp1)  -AHy 

        '''
        soft-thresholding the edges
        '''
        s = numpy.zeros_like(zz[pp]) # complex
        for pp in range(0,ndims):
            s_tmp[pp] = zz[pp] + bb[pp]

            s_r = numpy.hypot(s_tmp[pp].real, s_tmp[pp].imag)
            s = numpy.hypot(s_r, s.real)
#             s = numpy.sqrt(s**2 + s_tmp[pp]**2)
#             else:
#                 s =  s_tmp[pp]
#         s1 = zz[0] + bb[0]
#         s2 = zz[1] + bb[1]
#         s = sum((s_tmp[pp])**2 for pp in range(0,n_dims))
#         s = s1**2 + s2**2
        s += 1e-5

        threshold_value = 1/LMBD
        r =(s > threshold_value)*(s-threshold_value)/s#numpy.maximum(s - threshold_value ,  0.0)/s
        for pp in range(0,ndims):
            dd[pp] = s_tmp[pp]*r
#         dd[0] = s1*r
#         dd[1] = s2*r
#         df = zf+bf
#         
#         threshold_value=1.0/mu
#     
#         df.real =0.0+ (df.real>threshold_value)*(df.real - threshold_value) +(df.real<= - threshold_value)*(df.real+threshold_value)
#         df.imag = 0.0+(df.imag>threshold_value)*(df.imag - threshold_value) +(df.imag<= - threshold_value)*(df.imag+threshold_value) 
#                 df =     sy
        # end of shrinkage
        for pp in range(0,ndims):
            bb[pp] += zz[pp] - dd[pp]
#         bb[0] += zz[0] - dd[0] 
#         bb[1] += zz[1] - dd[1] 
#         bf += zf - df 
#                 self._update_b() # update b based on the current u

        AHyk = AHyk - zf # Linearized Bregman iteration f^k+1 = f^k + f - Au
#             print(outer)

    return xkp1 #(u,u_stack)

def _pipe_density(nufft, maxiter):
    '''
    Create the density function by iterative solution
    Generate pHp matrix
    '''
#         W = pipe_density(self.st['p'])
    # sampling density function
              
    W = numpy.ones(nufft.st['M'],dtype=nufft.dtype)
#         V1= self.st['p'].getH()
    #     VVH = V.dot(V.getH()) 
         
    for pp in range(0,maxiter):
#             E = self.st['p'].dot(V1.dot(W))

        E = nufft.forward(nufft.adjoint(W))
        W = (W/E)
   
    return W
 
def solve(nufft,   y,  solver=None, *args, **kwargs):
        """
        Solve NUFFT.
        The current version supports solvers = 'cg' or 'L1TVOLS' or 'L1TVLAD'.
        
        :param nufft: NUFFT_cpu object
        :param y: (M,) array, non-uniform data
        :return: x: image
            
        """
        if ('cgs' == solver) or ('qmr' ==solver) or ('minres'==solver):
            raise TypeError(solver +' requires real symmetric matrix')
        
    
        if None ==  solver:
    #         solver  =   'cg'
            print("solver must be one of the following solvers; \newline dc, cg, bicg, bicgstab, gmres, lgmres, L1OLS, L1LAD")
     
    
        if 'dc'   ==  solver:
            """
            Density compensation method
            nufft.st['W'] will be computed if doesn't exist
            If nufft.st['W'] exist then x2 = nufft.adjoint(nufft.st['W']*y)
            input: 
                y: (M,) array
            output:
                x2: Nd array
            """
            print(solver)
    #             try:
    # 
    #                 x = nufft.adjoint(nufft.st['W']*y)
    # 
    #             except: 
    
            W = _pipe_density( nufft, *args, **kwargs)
    
            x = nufft.adjoint(W*y)
    
            return x
        elif ('lsmr'==solver) or ('lsqr'==solver):
            """
            Assymetric matrix A
            Minimize L2 norm of |y-Ax|_2 or |y-Ax|_2+|x|_2
            Very stable 
            """
#                 A = nufft.sp
            def sp(k):
                k2 = k.reshape(nufft.Kd, order='C')
                return nufft.k2y(k2).ravel()
            def spH(y):
                y2 = y.reshape(nufft.st['M'], order='C')
                return nufft.y2k(y2).ravel()                
#                 return nufft.spH.dot(nufft.sp.dot(x))
#                 print('shape', (nufft.st['M']*nufft.batch, nufft.Kdprod*nufft.batch))
#                 print('spH ')
            A = scipy.sparse.linalg.LinearOperator((nufft.st['M']*nufft.batch, nufft.Kdprod*nufft.batch), matvec = sp, rmatvec = spH, )
            
            methods={'lsqr':scipy.sparse.linalg.lsqr,
                                'lsmr':scipy.sparse.linalg.lsmr,}
            k2 = methods[solver](A,  y.flatten(), *args, **kwargs)#,show=True)
            vec = k2[0]
            vec.shape = nufft.Kd
            xx = nufft.k2xx(vec)
            x= xx/nufft.sn
            return x#, k2[1:]        
        elif 'L1TVOLS' == solver:
            return  L1TVOLS(nufft, y, *args, **kwargs)
#         elif 'L1TVLAD' == solver:
#             return  L1TVLAD(nufft, y, *args, **kwargs)
        else:
            """
            Hermitian matrix A
            cg: conjugate gradient
            bicgstab: biconjugate gradient stablizing
            bicg: biconjugate gradient
            gmres: 
            lgmres:
            """
#             A = nufft.spHsp#nufft.st['p'].getH().dot(nufft.st['p'])
            def spHsp(x):
                k = x.reshape(nufft.Kd, order='C')
                return nufft.k2y2k(k).ravel()
#                 return nufft.spH.dot(nufft.sp.dot(x))
            
            A = scipy.sparse.linalg.LinearOperator((nufft.Kdprod*nufft.batch, nufft.Kdprod*nufft.batch), matvec = spHsp, rmatvec = spHsp, )

            
            methods={'cg':scipy.sparse.linalg.cg,   
                                 'bicgstab':scipy.sparse.linalg.bicgstab, 
                                 'bicg':scipy.sparse.linalg.bicg, 
    #                                  'cgs':scipy.sparse.linalg.cgs, 
                                 'gmres':scipy.sparse.linalg.gmres, 
                                 'lgmres':scipy.sparse.linalg.lgmres, 
    #                                  'minres':scipy.sparse.linalg.minres, 
    #                                  'qmr':scipy.sparse.linalg.qmr, 
                                 }
            k2 = methods[solver](A,  nufft.y2k(y).ravel(), *args, **kwargs)#,show=True)
    
    
            xx = nufft.k2xx(k2[0].reshape(nufft.Kd))
            x= xx/nufft.sn
            return x#     , k2[1:]       
