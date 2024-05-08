from pynufft import NUFFT
import numpy
import matplotlib.pyplot
import scipy.misc
import scipy.sparse.linalg

x0 = scipy.misc.ascent()[::2, ::2].astype(numpy.complex64)
# om = numpy.random.randn(256,2)

nu = numpy.pi*(numpy.arange(0, 512) - 256)/256
theta = numpy.arange(0, 360)*numpy.pi /180
om = numpy.empty((512, 360,2))
om[:,:,0] = numpy.einsum('a,b->ab', nu, numpy.cos(theta))
om[:,:,1] = numpy.einsum('a,b->ab', nu, numpy.sin(theta))
om = numpy.reshape(om, (512*360,2))
matplotlib.pyplot.plot(om[:,0], om[:,1],'x')
matplotlib.pyplot.show()

Nd = (256,256)
Kd = (512,512)
Jd = (6,6)

N = NUFFT()
N.plan(om, Nd, Kd, Jd)
y = N.forward(x0)
# A.x2xx(), A.xx2x() A.k2y() A.y2k(), A.xx2k(), A.k2xx(), 
m, n, prodk = om.shape[0],numpy.prod(Nd), numpy.prod(Kd)

# def 

def mv(v):
    out = numpy.zeros((m+m+prodk, ),dtype=numpy.complex64)
    out[:m] = 1*N.forward(v[:n].reshape(Nd))
    out[m:2*m] = N.k2y(v[n:].reshape(Kd))
    out[2*m:] =  N.xx2k(N.x2xx(v[:n].reshape(Nd))).flatten() - v[n:]
#     out[:m] = v[:m] + N.forward(v[m:].reshape(Nd))
#     out[m:] = N.adjoint(v[:m]).flatten()
    return out

def mvh(u):
    out =  numpy.zeros((n+prodk, ),dtype=numpy.complex64)
    out[:n] =1* N.adjoint(u[:m]).flatten() + ((1/N.sn)*N.k2xx(u[2*m:].reshape(Kd))).flatten()
    out[n:] = N.y2k(u[m:2*m]).flatten() - u[2*m:]
    
    
    return out
#     return N.forward(v.reshape(Nd))
# def mvh(y2):
#     x = N.adjoint(y2)
#     return x.flatten()

# Im = scipy.sparse.eye(om.shape[0],om.shape[0])
# In =  scipy.sparse.eye(numpy.prod(Nd),numpy.prod(Nd))*0

A = scipy.sparse.linalg.LinearOperator((m+m+prodk, n+prodk), matvec=mv, rmatvec=mvh)
y2 = numpy.zeros((2*m+prodk, ),dtype=numpy.complex64)
y2[:m] = y
y2[m:2*m] = y 
# y2[2*m:] *=0.0
# AT = scipy.sparse.linalg.LinearOperator((numpy.prod(Nd), om.shape[0]), matvec=mvh, rmatvec=mv, )
# A = scipy.sparse.linalg.aslinearoperator(A)
# AT = scipy.sparse.linalg.aslinearoperator(AT)
# Im = scipy.sparse.linalg.aslinearoperator(Im)
# In = scipy.sparse.linalg.aslinearoperator(In)
# print(Im.shape)
# print(In.shape)
# print(A.shape)
# print(AT.shape)

# A2 = scipy.sparse.bmat([[Im, A], [AT, None]])

# K = block((
#         (Im, A),
#         (AT, None)
#     ), arrtype=scipy.sparse.linalg.LinearOperator)

# x2 = scipy.sparse.linalg.lsmr(N,y,maxiter=40)[0].reshape(Nd)
# z = scipy.sparse.linalg.lsqr( A,y2, iter_lim=100, atol=1e-12)[0]
z = scipy.sparse.linalg.lsqr( A,y2, iter_lim=100, atol=1e-5)[0]
x2 = z[:n].reshape(Nd)
k = z[n:].reshape(Kd)
# print(numpy.linalg.norm(x2[:m]))

# x = scipy.
# 
# w = numpy.ones_like(y)
# for pp in range(0, 10):
#     print(pp)
#     w = w/N.forward(N.adjoint(w))
# 
# 
# 
# x2 = N.adjoint(y*w)
# x3 = N.solve(y, 'lsmr', maxiter=10)



matplotlib.pyplot.subplot(1,2,1)
matplotlib.pyplot.imshow(x2.real)
matplotlib.pyplot.subplot(1,2,2)
matplotlib.pyplot.imshow(numpy.log(abs(k)))
matplotlib.pyplot.show()





