
import numpy 
import matplotlib.pyplot as pyplot
from matplotlib import cm
gray = cm.gray
def indxmap_diff(Nd):
    """
    Preindixing for rapid image gradient ()
    
    Diff(x) = x.flat[d_indx[0]] - x.flat
    
    Diff_t(x) =  x.flat[dt_indx[0]] - x.flat
                            
    :param Nd: the dimension of the image
    :type Nd: tuple with integers
    :returns d_indx: image gradient
    :returns  dt_indx:  the transpose of the image gradient 
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
import scipy.sparse
def gradient_class(Nd, axis):
    d_indx, dt_indx = indxmap_diff(Nd)
    I = scipy.sparse.eye(numpy.prod(Nd), numpy.prod(Nd))
    data = numpy.ones((numpy.prod(Nd),))
    
    row_ind = d_indx[axis]
    col_ind= numpy.arange(0, numpy.prod(Nd)).astype(numpy.int) 
    G = scipy.sparse.csr_matrix(( data, 
                                            (row_ind, col_ind)), shape = (numpy.prod(Nd),numpy.prod(Nd))
                                 )
    G = G- I

    G = G.tocsr()
    return G
def gradient_class2(Nd, axis):
    d_indx, dt_indx = indxmap_diff(Nd)
    I = scipy.sparse.eye(numpy.prod(Nd), numpy.prod(Nd))
    data = numpy.ones((numpy.prod(Nd),))
    
    row_ind = dt_indx[axis]
    col_ind= numpy.arange(0, numpy.prod(Nd)).astype(numpy.int) 
    G = scipy.sparse.csr_matrix(( data, 
                                            (row_ind, col_ind)), shape = (numpy.prod(Nd),numpy.prod(Nd))
                                 )
    G = G- I

    G = G.tocsr()
    return G
def GBPDNA2(nufft, gy, maxiter, rho):
    """
    GBPDNA: test 3D total variation
    """
    import pynufft.src._helper.helper as helper
    f = gy
    def A(x):
        y2 = nufft.forward(x.reshape(nufft.st['Nd']))
        return y2
    
    def AH(y):
#         py = numpy.array(y.astype(numpy.complex64), order='C')
        x2 = nufft.adjoint(y).flatten()
        return x2    
    
    
    Nd = nufft.st['Nd']

#     Gz = gradient_class(Nd, 2)
    Gx = gradient_class2(Nd, 0)
    Gy = gradient_class2(Nd, 1)
    Gx2 = gradient_class2(Nd, 0)
    Gy2 = gradient_class2(Nd, 1)
#     Gz2 = gradient_class2(Nd, 2)
    Gxx = Gx2.dot(Gx2)
    Gyy = Gy2.dot(Gy2)
    
    M = nufft.st['M']
    v = numpy.ones(M,)

    for pp in range(0,20):
        w = A(AH((v)))
        lab = numpy.inner(w,numpy.conj(v))/numpy.inner(v,numpy.conj(v))
        tau_1 = 1/lab.real
    
        w = w/numpy.linalg.norm(w)
        v= w
    v= numpy.random.rand(numpy.prod(Nd),)

    for pp in range(0,20):
        w = Gx.getH().dot(Gx.dot(v))
        lab = numpy.inner(w,numpy.conj(v))/numpy.inner(v,numpy.conj(v))
        tau_2 = 1/(lab.real)
    
        w = w/numpy.linalg.norm(w)
        v= w
        
    print("tau_1 = ", tau_1)   
    print("tau_2 = ", tau_2)
#     tau_1 = 0.1*tau_1
    tau_2 = 0.1*tau_2*rho
#     tau_2 *= 3
    delta = 1.0
    mu = 0.001*numpy.max(numpy.abs(AH(f))[...])
    print("mu=",mu)
    
    def P_lambda(w_i, mu, tau_1):
        w_abs = numpy.abs(w_i)
    #     print(w_abs.shape)
    #     print(w_iw_abs.shape)
        out = ((w_i+1e-10)/(w_abs+1e-10))*mu/tau_1
        
        indx= w_abs <= (mu/tau_1)
        out[indx] =w_i[indx]
        return out
    def Q_f_eps(v, f, eps):
        v_f = v-f
        v_f_abs = numpy.abs(v_f)
        out = f + eps* v_f/v_f_abs
        indx = (v_f_abs <= eps)
        out[indx] = v[indx]
        return out
    N = numpy.prod(Nd)
    u_bold_k = numpy.zeros(N,)
    v_k = numpy.zeros(M,)
    z_k = numpy.zeros(M,)
    w_kx = numpy.zeros(N,)
    w_ky = numpy.zeros(N,)

    hx = numpy.zeros(N,)
    hy = numpy.zeros(N,)
#     hz = numpy.zeros(N,)
    hx2 = numpy.zeros(N,)
    hy2 = numpy.zeros(N,)
#     hz2 = numpy.zeros(N,)
    
    tmp_f=numpy.zeros(M,)
    
    eps = 1e-16
    for iter in range(0, maxiter):
        
        print(iter)
        tmp_u= u_bold_k - tau_1 * AH(v_k + tmp_f- z_k).flat[...]
        
        u_bar_kp1 = tmp_u   -    tau_1 *( Gx.getH().dot(w_kx) + Gy.getH().dot(w_ky) )
#                                           Gx2.getH().dot(w_kx2) + Gy2.getH().dot(w_ky2) + Gz2.getH().dot(w_kz2) ) 
        
    #     sx = Gx.dot(u_bar_kp1)
    #     sy = Gy.dot(u_bar_kp1)
    #     s = (sx**2 + sy**2)**0.5
          
        w_kp1x = P_lambda(w_kx + (tau_2/tau_1)*Gx.dot(u_bar_kp1), mu, tau_1)
        w_kp1y = P_lambda(w_ky+ (tau_2/tau_1)*Gy.dot(u_bar_kp1), mu, tau_1)
#         w_kp1z = P_lambda(w_kz+ (tau_2/tau_1)*Gz.dot(u_bar_kp1), mu, tau_1)
#         w_kp1x2 = P_lambda(w_kx2+ (tau_2/tau_1)*Gx2.dot(u_bar_kp1), mu, tau_1)
#         w_kp1y2 = P_lambda(w_ky2+ (tau_2/tau_1)*Gy2.dot(u_bar_kp1), mu, tau_1)
#         w_kp1z2 = P_lambda(w_kz2+ (tau_2/tau_1)*Gz2.dot(u_bar_kp1), mu, tau_1)
        
    #     hx = (sx+eps)/(s+eps)*Gx.getH().dot(w_kp1)
    #     hy = (sy+eps)/(s+eps)*Gy.getH().dot(w_kp1)
        
        u_bold_kp1 = tmp_u   -   tau_1 *( Gx.getH().dot(w_kp1x) + Gy.getH().dot(w_kp1y) )
#                                           Gx2.getH().dot(w_kp1x2) + Gy2.getH().dot(w_kp1y2) + Gz2.getH().dot(w_kp1z2)) 
        
        tmp_f=A(numpy.reshape( u_bold_kp1, Nd))
        z_kp1 = Q_f_eps(tmp_f + v_k, f, eps)
        v_kp1 = v_k + delta * (tmp_f    -   z_kp1)
        w_kx = w_kp1x
        w_ky = w_kp1y
#         w_kz = w_kp1z
#         w_kx2 = w_kp1x2
#         w_ky2 = w_kp1y2
#         w_kz2 = w_kp1z2
        
        u_bold_k = u_bold_kp1
        v_k = v_kp1
        z_k = z_kp1
    return numpy.reshape(u_bar_kp1, Nd)


import scipy.misc        
import pkg_resources
DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')   
# image = numpy.load(DATA_PATH +'phantom_3D_128_128_128.npz')['arr_0']#[0::2, 0::2, 0::2]
image = scipy.misc.ascent()[::2,::2]
# image = numpy.array(image, order='C')


Nd = (256,256) # time grid, tuple
Kd = (384,384) # frequency grid, tuple
Jd = (6,6) # interpolator 

#     om=       numpy.load(DATA_PATH+'om3D.npz')['arr_0']
numpy.random.seed(0)
om = numpy.random.randn(int(5e+5),2)

from pynufft import NUFFT, helper
# device = helper.device_list()[0]
NufftObj = NUFFT()

NufftObj.plan(om, Nd, Kd, Jd)


# NufftObj.offload(API = 'cuda',   platform_number = 0, device_number = 0)
# gx = NufftObj.thr.to_device(image.astype(numpy.complex64))
y =NufftObj.forward(image) 
import time
t0 = time.time()
restore_x2 = GBPDNA2(NufftObj, y, maxiter=5, rho = 2)
t1 = time.time()
restore_x = NufftObj.solve(y,'L1TVOLS', maxiter=50, rho=0.4)
t2 = time.time()
print("GBPDNA time = ", t1 - t0)
print("CG time = ", t2 - t1)

#restore_image1 = NufftObj.solve(kspace,'L1TVLAD', maxiter=300,rho=0.1)
# 
# restore_x2 = NufftObj.solve(gy,'L1TVOLS', maxiter=100,rho=0.2)
# tau_1 = 1
# tau_2 = 0.1


pyplot.subplot(1,3,1)
pyplot.imshow(image.real, label='original signal',cmap=gray)
pyplot.title('original')    
pyplot.subplot(1,3,2)
pyplot.imshow(numpy.abs(restore_x), label='L1TVLAD',cmap=gray)
pyplot.title('L1TVLAD')

pyplot.subplot(1,3,3)
pyplot.imshow(numpy.abs(restore_x2), label='GBPDNA',cmap=gray)
pyplot.title('GBPDNA (50 iterations)')
    

pyplot.show()


