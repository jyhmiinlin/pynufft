import numpy
def naive_DFT(x):
    N = numpy.size(x)
    X = numpy.zeros((N,),dtype=numpy.complex128)
    for m in range(0,N):    
        for n in range(0,N): 
            X[m] += x[n]*numpy.exp(-numpy.pi*2j*m*n/N)
    return X

x = numpy.random.rand(1024,)
# compute DFT
X=naive_DFT(x)
# compute FFT using numpy's fft function
X2 = numpy.fft.fft(x)
# now compare DFT with numpy fft
print('Is DFT close to fft?',numpy.allclose(X - X2,1e-12))
