# An example file of inverse Discrete Fourier transform (IDFT)
import numpy
def naive_IDFT(x):
    N = numpy.size(x)
    X = numpy.zeros((N,),dtype=numpy.complex128)
    for m in range(0,N):
        for n in range(0,N): 
            X[m] += x[n]*numpy.exp(numpy.pi*2j*m*n/N)
    return X/N
x = numpy.random.rand(1024,)
# compute FFT
X=numpy.fft.fft(x)
# compute IDFT using IDFT
x2 = naive_IDFT(X)
# now compare DFT with numpy fft
print('Is IDFT close to original?',numpy.allclose(x - x2,1e-12))
