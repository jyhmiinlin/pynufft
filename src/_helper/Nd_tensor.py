"""
Jyh-Miin Lin at UCL ICH/ Great Ormond Street Hospital
May - Aug 2018
Tensor family
Warning: Not for clinical purpose yet
"""
from __future__ import absolute_import
import numpy
import string
import scipy.sparse.linalg
alpha_list = list(string.ascii_lowercase)
abc_str = ()
for qq in range(0, 24):
    abc_str += (alpha_list[qq],)
    
fac_str= 'yz'
import matplotlib.pyplot
# import shrinkage_operator
# L1=shrinkage_operator.Shrinkage_L1()
numpy.random.seed(0)
mask = numpy.random.randn(128, 128,20) -  0.5
mask[mask<0] = 0
mask[mask>0] = 1
mask[54:74, 54:74, :] = 1

def DFT_matrix(N):
    i, j = numpy.meshgrid(numpy.arange(N), numpy.arange(N))
    omega = numpy.exp( - 2 * numpy.pi * 1J / N )
    W = numpy.power( omega, i * j ) / numpy.sqrt(N)
    return W

def tuple2om(om_tuple, ):

    last_axis = len(om_tuple)
#             print(last_axis, pp, Nd[pp])
#     if last_axis != Nd[pp]:
#         raise("Wrong value! the number of om should be equal to Nd[pp]!")    
    
    M = 0
    for jj in range(0, last_axis):
        M += om_tuple[jj].shape[0]
    dd = om_tuple[jj].shape[1] + 1
    
    om = numpy.ones((M, dd), dtype = numpy.float) 
    
    sumM = 0
    for jj in range(0, last_axis):
        len_om_tuple = om_tuple[jj].shape[0]    
#         for dimid in range(0, dd):
#             if ft_flag[dimid] is True:
        om[sumM:sumM+len_om_tuple, 0:dd-1] = om_tuple[jj]
#             else:
        om[sumM:sumM+len_om_tuple, dd-1] = jj
        sumM += len_om_tuple     
    return om


def S_lambda(x, lam):
#     return Lip.P(x, lam)
    shape = x.shape
    return L1.S(x,lam).reshape(shape, order='C')

# from Nd_nufft import tensor_nufft

class tensor_fft:
    
    def __init__(self, ft_axes, tensor_shape, undersampling=None):
        self.ft_axes =ft_axes
        self.tensor_shape = tensor_shape
#         if undersampling is True:

        self.undersampling = undersampling
        
    def forward(self, input_array):
        tmp1 = numpy.fft.fftshift(input_array, axes = self.ft_axes)
        tmp2 = numpy.fft.fftn(tmp1, axes = self.ft_axes)
        output_array = numpy.fft.fftshift(tmp2, axes = self.ft_axes)
        if self.undersampling is True:
            
#             try: 
            output_array = output_array * mask
        return output_array
    
    def adjoint(self, input_array):
#         print(input_array)
#         print("adjoint of tensor_fft",input_array.shape)
        if self.undersampling is True:
#             print('here')
#             try: 
            input_array = input_array * mask
        tmp1 = numpy.fft.ifftshift(input_array, axes = self.ft_axes)

#             except:
#                 self.mask = numpy.random.randn(input_array.shape[0],input_array.shape[1],input_array.shape[2]) + 1.5
#                 self.mask[self.mask<0] = 0
#                 self.mask[self.mask>0] = 1
#                 tmp1 = tmp1 * self.mask
        tmp2 = numpy.fft.ifftn(tmp1, axes = self.ft_axes)
        output_array = numpy.fft.ifftshift(tmp2, axes = self.ft_axes)
        return output_array
    
    def vectorize(self, input_array):
        vec = input_array.flatten(order='C')
        return vec
    
    def tensorize(self, input_vec):
        arr = numpy.reshape(input_vec, self.tensor_shape, order='C')
        return arr 
    
    def matvec(self, input_vec):
        """
        mathematicians' way
        equivalent to forward, but both the input and output are vectors
        """
        input_array = self.tensorize(input_vec)
        vec = self.vectorize(self.forward(input_array))
        return vec
    
    def rmatvec(self, input_vec):
        """
        mathematicians' way
        equivalent to adjoint, but both the input and output are vectors
        """        
        tensor = self.adjoint(self.tensorize(input_vec))
        vec = tensor.flatten(order='C')
        return vec
    def unfold(self, input_mat):
        """
        axes must be given!
        """
        new_shape = ()
        accumulate_shape = 1
        for pp in range(0, len(self.tensor_shape)):
            if pp in self.ft_axes: # the axis will be flattened if in axes
                accumulate_shape *= self.tensor_shape[pp]
            else:
                new_shape += (self.tensor_shape[pp], )
        new_shape = (accumulate_shape, ) + new_shape
        output_mat = numpy.reshape(input_mat, new_shape, order='C')
        return output_mat
    
    def fold(self, input_mat):

        arr = input_mat.reshape(self.tensor_shape, order='C')
        
        return arr 
    
    def TST(self, input_mat, lam):
#         self.hosvd(input_mat, rank=(3,3))
#         nuclear = self.forward(input_mat)
        nuclear = self.forward(input_mat)
#         nuclear = numpy.fft.fftshift(nuclear, axes = self.ft_axes)
#         print(nuclear.shape)
#         max_val = numpy.max(numpy.abs(nuclear.ravel()))
        nuclear = S_lambda(nuclear, lam)    #input_mat.shape[self.ft_axes[0]])
#         nuclear[:,:,3:-4,...] *=0
#         nuclear = numpy.fft.ifftshift(nuclear, axes = self.ft_axes)
        recovered = self.adjoint(nuclear)
        return recovered
    def TST0(self, input_mat, delta):
        """
        Apart from temporal FFT, adding temporal smoothing to enforce piece-wise constant 
        """
#         self.hosvd(input_mat, rank=(3,3))
#         nuclear = self.forward(input_mat)
        nuclear = numpy.fft.fftn(input_mat, axes = self.ft_axes)
#         nuclear[:,:,0:3,...] *= 100
#         nuclear = S_lambda(nuclear, lam)    #input_mat.shape[self.ft_axes[0]])
#         nuclear[:,:,-3:,...] /= 100
        if delta == 0:
            nuclear[:,:, 1:,...] *= 0
        else:
            nuclear[:,:,delta+1:-delta,...] *= 0 
        nuclear = numpy.fft.ifftn(nuclear, axes = self.ft_axes)
        return nuclear    
    def TST2(self, input_mat, delta):
        """
        Apart from temporal FFT, adding temporal smoothing to enforce piece-wise constant 
        """
#         self.hosvd(input_mat, rank=(3,3))
#         nuclear = self.forward(input_mat)
        nuclear = numpy.fft.fftn(input_mat, axes = self.ft_axes)
#         nuclear[:,:,0:3,...] *= 100
#         nuclear = S_lambda(nuclear, lam)    #input_mat.shape[self.ft_axes[0]])
#         nuclear[:,:,-3:,...] /= 100
        if delta == 0:
            nuclear[:,:, 1:,...] *= 0
        else:
            nuclear[:,:,delta+1:-delta,...] *= 0 
        nuclear = numpy.fft.ifftn(nuclear, axes = self.ft_axes)
        return nuclear
    
            
class htensor:
    """
    htensor: the Hermitian operator driven tensor
    Using the Higher order svd (HOSVD), Higher Order Orthogonal Iteration of Tensors (HOOI) and its
    Relation to PCA and GLRAM 
    htensor.hosvd(): 
    """
    def __init__(self):
        self.debug = False
        
    def hooi(self, A_array, rank=None, maxiter=2):
        self.shape = A_array.shape
        
        if rank is None:
            rank = A_array.shape
            
        if len(rank) != len(self.shape):
            raise("Wrong rank! rank must have the same length as the shape of the tensor")                    
        self.hosvd(A_array, rank=rank)
        U2 = list(self.U)
        shape = A_array.shape
        ndims = len(shape)
#         core = A_array
        for iter in range(0, maxiter):
#             core = A_array.copy()
            for pp in range(0, ndims):
                core = A_array.copy()
                for qq in range(0, ndims):
                    if qq != pp:
                        core = self.nmode( core, U2[qq], qq, if_conj=True)
                C = self.collapse(core, axis=pp)
                E1, XV1 = scipy.sparse.linalg.eigs(C*(1.0 + 0.0j), k=rank[pp])
                U2[pp] = XV1
        self.U = tuple(U2)


    def factor(self, A_array, jj, rank):
        B1 = self.collapse(A_array, jj)
        E1, XV1 = scipy.sparse.linalg.eigs(B1*(1.0 + 0.0j), k=rank)
        return XV1
        
    def hosvd(self, A_array, rank=None):
        """
        Find the factor matrices of A_array, default rank = A_array.shape
        """
        self.shape = A_array.shape
        
        if rank is None:
            rank = A_array.shape
            
        if len(rank) != len(A_array.shape):
            print(rank, A_array.rank)
            raise("Wrong rank! rank must have the same length as the shape of the tensor")            
    #     U, core = tucker.hosvd(T, rank=(192,192,50))
        shape = A_array.shape
        ndims = len(shape)
        U = []
        for jj in range(0, ndims):
#             B1 = self.collapse(A_array, jj)
#             E1, XV1 = scipy.sparse.linalg.eigs(B1, k=rank[jj])
            XV1 = self.factor(A_array, jj, rank[jj])
            U += [XV1, ]
        self.U = U
        self.rank = rank
#         self.core = self.dot(A_array)
    def SVT(self, input_mat, lam, rank=None):
        if rank != None:
            self.hosvd(input_mat, rank)
#         self.hooi(input_mat)
        nuclear = self.forward(input_mat)

#         max_val = numpy.max(numpy.abs(nuclear.ravel()))
        nuclear = S_lambda(nuclear, lam)
  
        recovered = self.adjoint(nuclear)
        return recovered
    def unfold(self, input_mat, axis):
        """
        axes must be given!
        """
        new_shape = ()
        accumulate_shape = 1
        for pp in range(0, len(self.tensor_shape)):
            if pp in self.axis: # the axis will be flattened if in axes
                new_shape += (self.tensor_shape[pp], )
            else:
                accumulate_shape *= self.tensor_shape[pp]
        new_shape = (accumulate_shape, ) + new_shape
        output_mat = numpy.reshape(input_mat, new_shape, order='C')
        return output_mat, original_shape
    
    def fold(self, input_mat, original_shape, axis):

        arr = input_mat.reshape(self.tensor_shape, order='C')
        
        return arr 
    def vectorize(self, array):
        return array.flatten(order='C')
    def tensorize(self, vec):
        try:
            arr = vec.reshape(self.shape, order='C')
        except:
            arr = vec.reshape(self.rank, order='C')
        return arr
    
    def collapse(self, A, axis):
        """
        reshape of A(flatten other axes except the given axis), then taking the self-adjoint of the flattened matrix
        
        """
        if self.debug:
            print('collapse')
        ndims = len(A.shape)
        if ndims > 24:
            raise('Wrong dimensions! Dimension should not exceed 24-D')
        in_str = ''
        out_str = ''    
        for jj in range(0, ndims):
            if jj == axis:
                in_str +=  'y'
                out_str +=  'z'
            else:
                in_str += abc_str[jj]
                out_str += abc_str[jj]
        einsum_instr = in_str +', ' + out_str + ' -> yz'
        if self.debug:
            print(nd, einsum_instr)
        B1 = numpy.einsum(einsum_instr,A,A.conj())
        return B1

    def allmode(self, input_mat, if_conj):
        """
        Restore the tensor from the input_mat(core).
        """
        ndims = len(input_mat.shape)
        output = input_mat
        for jj in range(0, ndims):
            output =  self.nmode(output, self.U[jj], jj, if_conj = if_conj)
        return output
    def forward(self, input_mat):
        output= self.allmode(input_mat, if_conj=True)
        return output
    def adjoint(self, input_mat):
        output= self.allmode(input_mat, if_conj=False)
        return output  
    
    def matvec(self, input_vec):
        input_mat = input_vec.reshape( self.shape, order='C')
        output_mat = self.forward(input_mat)
        return self.vectorize(output_mat)
    
    def rmatvec(self, input_vec):
        input_mat = input_vec.reshape( self.rank, order='C')
        output_mat = self.adjoint(input_mat)
        return self.vectorize(output_mat)
        
#     def dot(self, input_mat):
#         """
#         Project the tensor to nuclear form
#         Mode-N product of input_mat.
#         """        
#         ndims = len(input_mat.shape)
#         output = input_mat
#         for jj in range(0, ndims):
#             output =  self.nmode(output, self.U[jj], jj, if_conj = True) 
#             # Note: Taking the complex conjugate of factor matrices. Think about A = USV-> S = U' A V'
#         return output      

 
    def nmode(self, core, XV1, axis, if_conj):
        """
        Mode-1 product of tensor (core) and matrix (XV1)
        core: tensor
        XV1: factor matrx
        axis: the axis to be multiplied
        if_conj: taking the conjugate or not
        """
        ndims = len(core.shape)
        if ndims > 24:
            raise('Excessive dimensions! Dimension should not exceed 24-D')    
        
        """
        Now generate the string to control the Numpy.einsum
        the string is conncted after in_str and out_str have been known
        """
        in_str = ''
        out_str = ''    
        for jj in range(0, ndims):
            if jj == axis:
                in_str +=  'y'
                out_str +=  'z'
            else:
                in_str += abc_str[jj]
                out_str += abc_str[jj]
                
        if if_conj is True: # the dot operator
            """
            conjugate transpose to find the core 
            """
#             einsum_instr = 'yz'+',' +in_str +' ->  ' + out_str    
#             core = numpy.einsum(einsum_instr, XV1.conj(), core)
            einsum_instr = in_str + ',yz' +  ' ->  ' + out_str    
            core = numpy.einsum(einsum_instr, core,  XV1.conj())
            if self.debug:
                print(if_conj, einsum_instr)
        if if_conj is False: # the adjoint operator
#             einsum_instr = 'yz'+',' + out_str +' ->  ' + in_str    
#             core = numpy.einsum(einsum_instr, XV1, core)
            einsum_instr = out_str + ',yz'+' ->  ' + in_str    
            core = numpy.einsum(einsum_instr, core, XV1)
            if self.debug:        
                print(if_conj, einsum_instr)
        return core
    
def test_data(filename):
    xt_image = numpy.load('/tmp/'+filename+'.npy',)
    m = xt_image.shape[0]
    n = xt_image.shape[1]
    o = xt_image.shape[2]
    
    A = numpy.reshape(xt_image, (m,n,o))
    A = A/numpy.percentile(numpy.abs(A.ravel()), 95)
#     nuclear, recovered= tensor_compress(A,  (m, n, o), 1e-3)
    H = htensor()
#     H.hooi(A)
    H.hosvd(A, (9,9,9))
#     U = H.U
    
#     nuclear = A
    nuclear = H.allmode(A, if_conj=True)
    nuclear2 = nuclear.copy()

#     nuclear[numpy.abs(nuclear)  <   numpy.linalg.norm(nuclear)*threshold] = 0
    nuclear2[numpy.abs(nuclear)  <   5e-1] = 0
    recovered = H.allmode( nuclear, if_conj=False)    
#     nuclear = tensor_compress(nuclear)
    print(nuclear.shape)
    print(recovered.shape)
#     print(recovered)
    print('error of norm',numpy.linalg.norm(recovered - A)/numpy.linalg.norm( A))
    print('number of  nnz',numpy.count_nonzero(nuclear), ', Ratio = ',numpy.count_nonzero(nuclear)/ m/n/o)
#     import scipy.sparse
    for pp in range(0, 9):
        matplotlib.pyplot.subplot(3,3,pp +1)
        matplotlib.pyplot.imshow(numpy.abs(nuclear[:,:,pp]), )
    
    matplotlib.pyplot.show()
    for pp in range(0, 9):
        matplotlib.pyplot.subplot(3,3,pp +1)
        matplotlib.pyplot.imshow(numpy.abs(nuclear2[:,:,pp]), )
    
    matplotlib.pyplot.show()
    
#     matplotlib.pyplot.imshow(numpy.concatenate((numpy.real(recovered[:,:,0]),numpy.real(A[:,:,0])), axis = 1), cmap = matplotlib.cm.gray, vmin = 0, vmax = 1)
#     matplotlib.pyplot.show()
    
    
    from array2gif import write_gif
    
    xt_image = recovered #numpy.concatenate((numpy.abs(A), numpy.abs(recovered)/1e+2), axis = 0)
    print(xt_image.shape)
    xt_image = 255*numpy.abs(xt_image)/numpy.percentile(numpy.abs(xt_image[:,:,...].flatten()), 95)
    xt_image[xt_image >255] = 255
    import scipy.misc
    data = ()
    for jj in range(0, o):
        data += (numpy.tile(xt_image[:,:,jj].astype(numpy.int32),(3,1,1)), )
    write_gif(data, '/tmp/'+filename+'recovered.gif', fps=1000/o)    
    
    xt_image = A
    print(xt_image.shape)
    xt_image = 255*numpy.abs(xt_image)/numpy.percentile(numpy.abs(xt_image[:,:,...].flatten()), 95)
    xt_image[xt_image >255] = 255
    import scipy.misc
    data = ()
    for jj in range(0, o):
        data += (numpy.tile(xt_image[:,:,jj].astype(numpy.int32),(3,1,1)), )
    write_gif(data, '/tmp/'+filename+'tensor_origin.gif', fps=1000/o)         

def test_2D():
    import scipy.misc
    A = scipy.misc.ascent()*(1+1.0j)
    print(A.shape)
    
    H = htensor()
    
    H.hosvd(A, (50,50))
#     H.hooi(A, (50,50), maxiter=200)
    
#     core = H.forward(A)
    core = H.matvec(A.flatten())

#     H.hosvd(A, (30,30))
#     H.hosvd(A, (100,100))
#     
# #     core = H.forward(A)
#     core2 = H.matvec(A.flatten())
#     
#     print(core[0:2], core2[0:2])
#     matplotlib.pyplot.imshow(numpy.abs(core[:,:]))
#     matplotlib.pyplot.show()    
#     core[numpy.abs(core)<1e+3] = 0
    
#     core = (core/numpy.abs(core))*core**2/numpy.sqrt(core**2 + 1)
    
    print('compression ratio', numpy.count_nonzero(A)/numpy.count_nonzero(core) )
#     recovered = H.adjoint(core)
    recovered = H.tensorize(H.rmatvec(core))
    
#     y = H.matvec(input_vec)
    
    matplotlib.pyplot.imshow(recovered.imag)
    matplotlib.pyplot.show()
    
def test_12D():
    import scipy.misc
    A = numpy.random.randn(3,4,5,3,4,5,3,4,5,3,4,5) + 1.0j * numpy.random.randn(3,4,5,3,4,5,3,4,5,3,4,5)
    print(A.shape)
    
    H = htensor()
    
    H.hosvd(A, (3,4,5,3,4,5,3,4,5,3,4,5))
#     XT = H.unfold(A, (0, 1, 2, 3))
#     print("XT.shape",XT.shape)
    core = H.forward(A)
    
    print(core.imag)
    core[numpy.abs(core)<1e-1 * numpy.max(numpy.abs(core))] = 0
    
#     core = (core/numpy.abs(core))*core**2/numpy.sqrt(core**2 + 1)
    recovered = H.adjoint(core)    
    print('compression ratio', numpy.count_nonzero(A)/numpy.count_nonzero(core), ' MSE=', numpy.linalg.norm(A- recovered)/numpy.linalg.norm(A) )
 
def test_FFT():
    import scipy.misc
#     A = scipy.misc.ascent()*(1+1.0j)
    A = numpy.random.randn(14,14)
    print(A.shape)
    
    H = htensor()
    H.U=()
    for pp in range(0, len(A.shape)):
        H.U += (DFT_matrix(A.shape[pp]).conj().T,)
        
    H.shape = A.shape
    H.rank = A.shape
#     H.hosvd(A, (50,50))
#     H.hooi(A, (50,50), maxiter=200)
    
#     core = H.forward(A)
    core =H.tensorize( H.matvec(A))    
#     core =H.tensorize( H.rmatvec(core.flatten()))    
    core = numpy.fft.ifftn(core) * (numpy.prod(A.shape)**0.5)
    print(core.shape)
    print(numpy.linalg.norm(core - A))
#     matplotlib.pyplot.imshow(numpy.log(numpy.abs(core[:,:])))
#     matplotlib.pyplot.imshow(core.real)
#     matplotlib.pyplot.show()

if __name__ == '__main__':
#     test_FFT()
#     matplotlib.pyplot.imshow(mask[:,:,0], cmap = matplotlib.pyplot.cm.gray)
#     matplotlib.pyplot.show()
#     test_2D()
    test_12D()
    
#     for filename in (#'rho_meas_MID30_Adult_bSFFP_radial_RT_FID546794.dat.mat',
#                     #'rho_meas_MID33_Adult_bSFFP_radial_RT_FID546797.dat.mat', 
#                     'rho_meas_MID34_Adult_bSFFP_radial_RT_FID546798.dat.mat',):
#                     #'rho_meas_MID35_Adult_bSFFP_radial_RT_FID546799.dat.mat'):
#         pass
#         test_data(filename)


#!/usr/bin/env python

