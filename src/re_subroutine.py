"""
Metaprogramming subroutines (using reikna, pyopencl, pycuda)
========================================================
"""

from __future__ import absolute_import # Python2 compatibility
# from . import cSelect

def create_kernel_sets(API):
    """
    Create the kernel from the kernel sets.
    Note that in some tests (Benoit's and my tests) CUDA shows some degraded accuracy. 
    This loss of accuracy was due to undefined shared memory behavior, which I don't fully understand.
    This has been fixed in 2019.2.0 as the operations are moved to global memory.
    """
    kernel_sets = ( cMultiplyScalar() + 
                        cCopy() + 
                        cTensorCopy() +
                        cTensorMultiply() + 
                        cMultiplyVec() +
                        cHypot() +  
                        cAddScalar() + 
                        cSelect() + 
                        cAddVec() +  
                        cMultiplyVecInplace() + 
                        cMultiplyConjVecInplace() +
                        cMultiplyRealInplace() + 
                        cMultiplyConjVec() + 
                        cDiff() + 
                        cSqrt() + 
                        cAnisoShrink() +  
                        cRealShrink() +
                        cSpmv() + 
                        cSpmvh() + 
                        cCopyColumn()+
                        cHadamard())
#     if 'cuda' is API:
#         print('Select cuda interface')
#         kernel_sets =  atomic_add.cuda_add + kernel_sets
#     elif 'ocl' is API:
#         print("Selecting opencl interface")
#         kernel_sets =  atomic_add.ocl_add + kernel_sets
    kernel_sets = atomic_add(API) + kernel_sets
    return kernel_sets

def cMultiplyScalar():
    """
    Return the kernel source for cMultiplyScalar.
    """
    code_text ="""
        KERNEL void cMultiplyScalar(
                     const float2 CA,
                GLOBAL_MEM float2 *CX)
        { 
        // Scale CX by CA: CX=CA*CX
        //  CA: scaling factor(float2)
        //*CX: input, output array(float2)
        unsigned long gid = get_global_id(0);  
        CX[gid].x=CA.x*CX[gid].x-CA.y*CX[gid].y;
        CX[gid].y=CA.x*CX[gid].y+CA.y*CX[gid].x;
        };           
        """
    return code_text
def cCopyColumn():
    code_text = """
    KERNEL void cCopyColumn( 
            const unsigned int n, // start
            const unsigned int m, // step
             GLOBAL_MEM  const float2 *CX,
             GLOBAL_MEM             float2 *CY)
    {
    // Copy x to y: y = x;
    //CX: input array (float2)
    // CY output array (float2)
    unsigned long gid=get_global_id(0);  
    CY[gid]=CX[n+m*gid];
    };
    """  
    return code_text    
def cCopy():
    """
    Return the kernel source for cCopy
    """
    code_text = """
    KERNEL void cCopy( 
             GLOBAL_MEM  const float2 *CX,
             GLOBAL_MEM             float2 *CY)
    {
    // Copy x to y: y = x;
    //CX: input array (float2)
    // CY output array (float2)
    unsigned long gid=get_global_id(0);  
    CY[gid]=CX[gid];
    };
    """  
    return code_text

def cTensorCopy():
    """
    Return the kernel source for cTensorCopy.
    """
    code_text = """
    KERNEL void cTensorCopy(
        const unsigned int batch, 
        const unsigned int dim,
        GLOBAL_MEM const  unsigned int *Nd_elements,
        GLOBAL_MEM const  unsigned int *Kd_elements,
        GLOBAL_MEM const  float *invNd,
        GLOBAL_MEM const float2 *indata,
        GLOBAL_MEM       float2 *outdata,
        const int direction)
    {
    
    const unsigned int gid=get_global_id(0); 
    
    unsigned int curr_res = gid;
    unsigned int new_idx = 0;
    unsigned int group;
    
    for (unsigned int dimid =0; dimid < dim; dimid ++){
        group = (float)curr_res*invNd[dimid];
        new_idx += group * Kd_elements[dimid];
        curr_res = curr_res - group * Nd_elements[dimid];
    };
    
    if (direction == 1) {
        for (unsigned int bat=0; bat < batch; bat ++ )
        {
            outdata[new_idx*batch+bat]= indata[gid*batch+bat];
         };   
    };
    
    if (direction == -1) {
        for (unsigned int bat=0; bat < batch; bat ++ )
        {
            outdata[gid*batch+bat]= indata[new_idx*batch+bat];
        };   
    };
                   
                   
    };
    """
    return code_text  

def cTensorMultiply():
    """
    Return the kernel source for cTensorMultiply
    """
    code_text = """
        KERNEL void cTensorMultiply(
            const unsigned int batch, // batch 
            const unsigned int dim, // dimensions
            GLOBAL_MEM const  unsigned int *Nd, // In batch mode, Nd*batch but is calculated outside the kernel
            GLOBAL_MEM const  unsigned int *Nd_elements, // Number of elements to move along the dimension = strides / itemsize
            GLOBAL_MEM const  float *invNd_elements,   //  float: inverse of the Nd_elements, which aims for fast division // batch mode: Nd_elements / batch
            GLOBAL_MEM const float *vec, // Real, vector, length sum Nd[dimid]
            GLOBAL_MEM       float2 *outdata, 
            const unsigned int div) 
        {
        const unsigned int gid=get_global_id(0); 
        const unsigned int pid = (float)gid / (float)batch;
        // const unsigned int bat = gid - pid * batch;
        
        unsigned int group;
        unsigned int Nd_indx_shift = 0;
        float mul = 1.0; 
        unsigned int res = pid; 
        
        for (unsigned int dimid = 0; dimid < dim; dimid ++){
            group = (float)res * invNd_elements[dimid]; // The index along the axis
            res = res - group * Nd_elements[dimid];
            
            const unsigned int N = Nd[dimid]; 
            
            mul = mul * vec[group + Nd_indx_shift];
            
            Nd_indx_shift = Nd_indx_shift + N;
        }
        
        if (div == 1){
            // for (unsigned int bat = 0; bat < batch; bat ++ )
            // {
            float2 tmp = outdata[gid];
            tmp.x = tmp.x /  mul;
            tmp.y = tmp.y / mul;
            outdata[gid] = tmp;
            // };
            };
        if (div == 0){
           // for (unsigned int bat = 0; bat < batch; bat ++ )
            // {
            float2 tmp = outdata[gid];
            tmp.x = tmp.x *  mul;
            tmp.y = tmp.y * mul;
            outdata[gid] = tmp;
            // };
            };    
                       
        };
        """  
    return code_text

def atomic_add(API):
    """
    Return the atomic_add for the given API. 
    Overcome the missing atomic_add_float for OpenCL-1.2. 
    Note: will be checked if OpenCL 2.0 provided by all GPU vendors. 
    """
    
    ocl_add = """
    // #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
 
    KERNEL void atomic_add_float2_typen( 
            GLOBAL_MEM float2 *ptr, 
            const float2 temp) 
        { // Add atomic two sum for real/imaginary parts
        
        __global const float * realptr = ( __global float *)ptr;
        __global const float * imagptr = ( __global float *)ptr + 1;
        float prevVal;
        float newVal;
        do { // atomic add of the value
            prevVal = *realptr ;
            newVal = prevVal + temp.x;
        } while (atomic_cmpxchg((__global volatile unsigned int *)realptr, as_int(prevVal), as_int(newVal)) != as_int(prevVal));
 
        // End of real part
        
        // Now do the imaginary part
        
        do { // atomic add of the value
            prevVal = *imagptr ;
            newVal = prevVal + temp.y;
        } while (atomic_cmpxchg((__global volatile unsigned int *)imagptr, as_int(prevVal), as_int(newVal)) != as_int(prevVal));
 
    };   
 
    KERNEL void atomic_add_float2( 
            GLOBAL_MEM float2 *ptr, 
            const float2 temp) 
        { // Add atomic two sum for real/imaginary parts
        union {
            unsigned int intVal;
            float floatVal;
        } newVal;
        
        union {
            unsigned int intVal;
            float floatVal;
        } prevVal;
 
            
        __global const float * realptr = ( __global float *)ptr;
        __global const float * imagptr = ( __global float *)ptr + 1;
          
        do { // atomic add of the value
            prevVal.floatVal = *realptr ;
            newVal.floatVal = prevVal.floatVal + temp.x;
        } while (atomic_cmpxchg((__global volatile unsigned int *)realptr, prevVal.intVal, newVal.intVal) != prevVal.intVal);
 
        // End of real part
        
        // Now do the imaginary part
        
        do {
            prevVal.floatVal = *imagptr;
            newVal.floatVal = prevVal.floatVal + temp.y;
        } while (atomic_cmpxchg((__global volatile unsigned int *)imagptr, prevVal.intVal, newVal.intVal) != prevVal.intVal);
   
    };  
    
    
    """
    
    cuda_add = """
    
    __device__ void atomic_add_float2( 
            GLOBAL_MEM float2 * ptr, 
            const float2 temp) 
    { // Wrapper around CUDA atomicAdd();
    atomicAdd((float*)ptr, temp.x);
    atomicAdd((float*)ptr+1, temp.y);
    // atomic_add_float_twosum((float*)ptr, temp.x, (float*)res);
    // atomic_add_float_twosum((float*)ptr+1, temp.y, (float*)res+1);
    };       
    
    """  

    if API == 'cuda':
        code_text = cuda_add
    elif API == 'ocl':
        code_text = ocl_add
    return code_text
    
def cMultiplyVec():
    """
    Return the kernel source of cMultiplyVec
    """
    code_text="""
        KERNEL void cMultiplyVec( GLOBAL_MEM float2 *a,
                                            GLOBAL_MEM float2 *b,
                                            GLOBAL_MEM float2 *dest)
        {    const unsigned int i = get_global_id(0);
            dest[i].x = a[i].x*b[i].x-a[i].y*b[i].y;
            dest[i].y = a[i].x*b[i].y+a[i].y*b[i].x;
            //barrier(CLK_GLOBAL_MEM_FENCE); 
        };
        """    
    return code_text

def cAddScalar():
    """
    Return the kernel source for cAddScalar.
    """
    code_text ="""
        KERNEL void cAddScalar(const float2 CA,
                                            GLOBAL_MEM float2 *CX)
        { 
        // (single complex) scale x by a: x = x + ca;
        // CA: add factor 
        // CX: input and output array (float2)
        int gid = get_global_id(0);  
        CX[gid].x += CA.x;
        CX[gid].y += CA.y;
        };
        """
    return code_text

def cHadamard():
    """
    Return the Hadamard operations related kernel sources.
    """
    
    R="""
    // Batched copying indata[order1] to outdata[order2] 
    // Superceded by cTensorCopy()
    // However left here as a general function.
        KERNEL void cSelect2( 
        const unsigned int Reps, 
        GLOBAL_MEM const  unsigned int *order1,
        GLOBAL_MEM const  unsigned int *order2,
        GLOBAL_MEM const float2 *indata,
        GLOBAL_MEM       float2 *outdata)
        {
        const unsigned int gid=get_global_id(0); 
        const unsigned int t = ((float)gid / (float)Reps); // indptr 
        const unsigned int r = gid - t*Reps; // residue
        
        const unsigned int index2 = order2[t]*Reps + r;
        const unsigned int index1 = order1[t]*Reps + r;
        
        outdata[index2]=indata[index1];
        };
        
        KERNEL void cDistribute(    
        const    unsigned int    Reps,
        const    unsigned int    prodNd, 
        GLOBAL_MEM const float2 *arr_large, // sensitivity and scaling array, prodNd*Reps
        GLOBAL_MEM const float2 *arr_image, // image array, prodNd
        GLOBAL_MEM float2 *arr_out // output array, prodNd * Reps
        ) 
        {  
        const unsigned int t = get_global_id(0);
        //const unsigned int nd = t/Reps;
        const unsigned int nd = ((float)t / (float)Reps);
        if (nd < prodNd){
        const float2 u = arr_large[t]; 
        const float2 v = arr_image[nd];
        float2 w; 
        w.x = u.x * v.x - u.y * v.y; 
        w.y = u.x * v.y + u.y * v.x; 
        arr_out[t] = w;
        }
        };  // End of cDistribute
        
        
        
        KERNEL void cMerge(
        const    unsigned int    Reps,
        const    unsigned int    prodNd, 
        GLOBAL_MEM const float2 *arr_large, // sensitivity and scaling array, prodNd*Reps
        GLOBAL_MEM const float2 *arr_image, // image array, prodNd*Reps
        GLOBAL_MEM float2 *arr_out // reduced output array, prodNd 
        )
        {   
        const unsigned int t = get_local_id(0);
        //const float float_reps = (float)Reps;
        const unsigned int vecWidth=${LL};
        // Thread ID within wavefront
        const unsigned int id = t & (vecWidth-1);
        // One row per wavefront
        unsigned int vecsPerBlock=get_local_size(0)/vecWidth;
        unsigned int myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth);
        LOCAL_MEM float2 partialSums[${LL}];
        float2 zero;
        zero.x = 0.0;
        zero.y = 0.0;
        
        partialSums[t] = zero;
        // float2  y= zero;
        if (myRow < prodNd)
        {
        const unsigned int vecStart = myRow * Reps;
        const unsigned int vecEnd = vecStart + Reps;            
        for (unsigned int j = vecStart+id;  j<vecEnd; j += vecWidth)
        {
            
              const float2 u = arr_large[j]; // sensitivities and scaling, complex 
              const float2 v = arr_image[j];  
              float2 w; 
              w.x = u.x * v.x + u.y * v.y; // conjugate of u 
              w.y = u.x * v.y - u.y * v.x; 
              w.x = w.x/(float)Reps;
              w.y = w.y/(float)Reps;
            partialSums[t] = w;        //partialSums[t] + y;
        }
        
        LOCAL_BARRIER; 
        //__syncthreads();
        //barrier(CLK_LOCAL_MEM_FENCE);
        // Reduce partial sums
        unsigned int bar = vecWidth / 2;
        while(bar > 0)
        {
             if (id < bar)
            partialSums[t] = partialSums[t] + partialSums[t+bar];
             
             //barrier(CLK_LOCAL_MEM_FENCE);
             //__syncthreads();
             LOCAL_BARRIER;
             bar = bar / 2;
        }            
        // Write result 
        if (id == 0)
        {
         arr_out[myRow]=partialSums[t]; 
        }
        }
        };    // End of cMerge
        
        KERNEL void cAggregate(
        const    unsigned int    Reps,
        const    unsigned int    prodNd, 
        GLOBAL_MEM const float2 *arr_image, // image array, prodNd*Reps
        GLOBAL_MEM float2 *arr_out // reduced output array, prodNd 
        )
        {   
        const unsigned int t = get_local_id(0);
        //const float float_reps = (float)Reps;
        const unsigned int vecWidth=${LL};
        // Thread ID within wavefront
        const unsigned int id = t & (vecWidth-1);
        // One row per wavefront
        unsigned int vecsPerBlock=get_local_size(0)/vecWidth;
        unsigned int myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth);
        LOCAL_MEM float2 partialSums[${LL}];
        float2 zero;
        zero.x = 0.0;
        zero.y = 0.0;
        
        partialSums[t] = zero;
        // float2  y= zero;
        if (myRow < prodNd)
        {
        const unsigned int vecStart = myRow * Reps;
        const unsigned int vecEnd = vecStart + Reps;            
        for (unsigned int j = vecStart+id;  j<vecEnd; j += vecWidth)
        {
          float2 v = arr_image[j];
          v.x = v.x/(float)Reps;
          v.y = v.y/(float)Reps;
          partialSums[t] =partialSums[t] + v;        //partialSums[t] + y;
        }
        
        LOCAL_BARRIER; 
        //__syncthreads();
        //barrier(CLK_LOCAL_MEM_FENCE);
        // Reduce partial sums
        unsigned int bar = vecWidth / 2;
        while(bar > 0)
        {
             if (id < bar)
            partialSums[t] = partialSums[t] + partialSums[t+bar];
             
             //barrier(CLK_LOCAL_MEM_FENCE);
             //__syncthreads();
             LOCAL_BARRIER;
             bar = bar / 2;
        }            
        // Write result 
        if (id == 0)
        {
         arr_out[myRow]=partialSums[t]; 
        }
        }
        LOCAL_BARRIER;
        };    // End of cAggregate
        
        KERNEL void cPopulate(    
        const    unsigned int    Reps,
        const    unsigned int   prodNd, 
        GLOBAL_MEM const float2 *arr_image, // image array, prodNd
        GLOBAL_MEM float2 *arr_out // output array, prodNd * Reps
        ) 
        {  
        const unsigned int t = get_global_id(0);
        //const unsigned int nd = t/Reps;
        const unsigned int nd = ((float)t / (float)Reps);
        if (nd < prodNd){
        const float2 v = arr_image[nd];
        arr_out[t] = v;
        }
        LOCAL_BARRIER;
        };  // End of cPopulate    
        
        """
    return R
def cSpmvh():
    """
    Return the cSpmvh related kernel source. 
    Only pELL_spmvh_mCoil is provided for Spmvh.
    NUFFT_hsa_legacy reuses the cCSR_spmv() function, which doubles the storage. 
    """
    
    R="""
    
        KERNEL void pELL_spmvh_mCoil(
        const    unsigned int    Reps,             // number of coils
        const    unsigned int    nRow,        // number of rows
        const    unsigned int    prodJd,     // product of Jd
        const    unsigned int    sumJd,     // sum of Jd
        const    unsigned int    dim,           // dimensionality
        GLOBAL_MEM const unsigned int *Jd,    // Jd
        // GLOBAL_MEM const unsigned int *curr_sumJd,    // 
        GLOBAL_MEM const unsigned int *meshindex,    // meshindex, prodJd * dim
        GLOBAL_MEM const unsigned int *kindx,    // unmixed column indexes of all dimensions
        GLOBAL_MEM const float2 *udata,    // interpolation data before Kronecker product
        GLOBAL_MEM    float2    *k, 
        //GLOBAL_MEM float2 *res,
        GLOBAL_MEM const float2 *input)   // y
        {      
        const unsigned int t = get_local_id(0);
        const unsigned int vecWidth=${LL};
        // Thread ID within wavefront
        const unsigned int id = t & (vecWidth-1);
        
        // One row per wavefront
        unsigned int vecsPerBlock=get_local_size(0)/vecWidth;
        unsigned int myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth); // the myRow-th non-Cartesian sample
        unsigned int m = myRow / Reps;
        unsigned int nc = myRow - m * Reps;
        
        float2 zero;
        zero.x = 0.0;
        zero.y = 0.0;
        
        
        if (myRow < nRow * Reps)
        {
        const unsigned int vecStart = 0; 
        const unsigned int vecEnd =prodJd;             
        float2  u=zero;
        
        for (unsigned int j = vecStart+id;  j<vecEnd; j += vecWidth)
                        {    
                        // now doing the first dimension
                        unsigned int index_shift = m * sumJd;
                        // unsigned int tmp_sumJd = 0;
                        unsigned int J = Jd[0];
                        unsigned int index =    index_shift +  meshindex[dim*j + 0];
                        unsigned int col = kindx[index] ;
                        float2 spdata = udata[index];
                        index_shift += J; 
                        for (unsigned int dimid = 1; dimid < dim; dimid ++ )
                                {
                                J = Jd[dimid];
                                index =   index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
                                col += kindx[index];// + 1  ;                                            // the column index of the current j
                                float tmp_x = spdata.x;
                                float2 tmp_udata = udata[index];
                                spdata.x = tmp_x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
                                spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
                                index_shift  += J;
                                }; // Iterate over dimensions 1 -> Nd - 1
                        
                        float2 ydata=input[myRow]; // kout[col];
                        u.x =  spdata.x*ydata.x + spdata.y*ydata.y;
                        u.y =  - spdata.y*ydata.x + spdata.x*ydata.y;
                        
                        atomic_add_float2(k + col*Reps + nc, u);//, res + col*Reps + nc);
                        LOCAL_BARRIER;
                        // atomic_add_float2(k + col*Reps + nc, u, res + col*Reps + nc);
                        }; // Iterate for (unsigned int j = 0;  j  <  prodJd; j ++)
        };  // if (m < nRow)
        
        };    // End of xELL_spmvh_mCoil    
    
        
        KERNEL void pELL_spmvh_mCoil_new(
        const    unsigned int    Reps,             // number of coils
        const    unsigned int    nRow,        // number of rows
        const    unsigned int    prodJd,     // product of Jd
        const    unsigned int    sumJd,     // sum of Jd
        const    unsigned int    dim,           // dimensionality
        GLOBAL_MEM const unsigned int *Jd,    // Jd
        // GLOBAL_MEM const unsigned int *curr_sumJd,    // 
        GLOBAL_MEM const unsigned int *meshindex,    // meshindex, prodJd * dim
        GLOBAL_MEM const unsigned int *kindx,    // unmixed column indexes of all dimensions
        GLOBAL_MEM const float2 *udata,    // interpolation data before Kronecker product
        GLOBAL_MEM    float2    *k, 
        GLOBAL_MEM    float2    *res,
        GLOBAL_MEM const float2 *input)   // y
        {
        unsigned int myRow0= get_global_id(0);
        unsigned int myRow= myRow0/(float)Reps;
        unsigned int nc = myRow0 - myRow*Reps;
        float2 zero;
        zero.x = 0.0;
        zero.y = 0.0;
        if (myRow < nRow){ 
            for (unsigned int j = 0;  j  <  prodJd; j ++){
                    float2 u = zero;

                    // now doing the first dimension
                    unsigned int index_shift = myRow * sumJd;
                    // unsigned int tmp_sumJd = 0;
                    unsigned int J = Jd[0];
                    unsigned int index =    index_shift +  meshindex[dim*j + 0];
                    unsigned int col = kindx[index] ;
                    float2 spdata = udata[index];
                    index_shift += J; 
                    for (unsigned int dimid = 1; dimid < dim; dimid ++ ){
                            J = Jd[dimid];
                            index =   index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
                            col += kindx[index];// + 1  ;                                            // the column index of the current j
                            float tmp_x = spdata.x;
                            float2 tmp_udata = udata[index];
                            spdata.x = tmp_x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
                            spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
                            index_shift  += J;
                    }; // Iterate over dimensions 1 -> Nd - 1
                    
                    float2 ydata=input[myRow*Reps + nc]; // kout[col];
                    u.x =  spdata.x*ydata.x + spdata.y*ydata.y;
                    u.y =  - spdata.y*ydata.x + spdata.x*ydata.y;
                    atomic_add_float2(k + col*Reps + nc, u);
                        
                }; // Iterate for (unsigned int j = 0;  j  <  prodJd; j ++)
                
            };  // if (m < nRow)
        
        };    // End of pELL_spmvh_mCoil          
        """
    return R

def cHypot():
    """
    Return the kernel code for hypot, which computes the sqrt(x*x + y*y) without intermediate overflow.
    """
    
    R="""
    KERNEL void cHypot(GLOBAL_MEM float2 *x,
                                    GLOBAL_MEM const float2 *y)
    {
    const unsigned int gid =  get_global_id(0); 
    float2 tmp_x;
    float2 tmp_y;
    tmp_x = x[gid];
    tmp_y = y[gid];
    tmp_x.x = hypot( tmp_x.x, tmp_x.y); // sqrt( tmp_x.x*tmp_x.x + tmp_x.y*tmp_x.y);
    tmp_y.x = hypot( tmp_y.x, tmp_y.y); // sqrt( tmp_y.x*tmp_y.x + tmp_y.y*tmp_y.y);
    
    x[gid].x = hypot(tmp_x.x, tmp_y.x);
    x[gid].y = 0.0;
    
    };
    """    
    return R

def cSpmv():
    """
    Return the kernel sources for cSpmv related operations,
    providing cCSR_spmv_vector and cpELL_spmv_mCoil.
    """
    
    R = """
        KERNEL void cCSR_spmv_vector(    
        const     unsigned int   Reps,            // Number of coils
        const    unsigned int    numRow,
        GLOBAL_MEM const unsigned int *rowDelimiters, 
        GLOBAL_MEM const unsigned int *cols,
        GLOBAL_MEM const float2 *val,
        GLOBAL_MEM const float2 *vec, 
        GLOBAL_MEM float2 *out)
        {   
        const unsigned int t = get_local_id(0);
        const unsigned int vecWidth=${LL};
        // Thread ID within wavefront
        const unsigned int id = t & (vecWidth-1);
        // One row per wavefront
        unsigned int vecsPerBlock=get_local_size(0)/vecWidth;
        unsigned int myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth);
        unsigned int m = myRow / Reps;
        unsigned int nc = myRow - m * Reps;        
        LOCAL_MEM float2 partialSums[${LL}];
        float2 zero;
        zero.x = 0.0;
        zero.y = 0.0;
        
        partialSums[t] = zero;
        float2  y= zero;
        
        if (myRow < numRow * Reps)
        {
        const unsigned int vecStart = rowDelimiters[m];
        const unsigned int vecEnd = rowDelimiters[m+1];            
        for (unsigned int j = vecStart+id;  j<vecEnd; j += vecWidth)
        {
        const unsigned int col = cols[j];
        const float2 spdata=val[j];
        const float2 vecdata=vec[col * Reps + nc];                        
        y.x=spdata.x*vecdata.x - spdata.y*vecdata.y;
        y.y=spdata.y*vecdata.x + spdata.x*vecdata.y;
        partialSums[t] = partialSums[t] + y;
        }
        
        LOCAL_BARRIER; 
        //__syncthreads();
        //barrier(CLK_LOCAL_MEM_FENCE);
        // Reduce partial sums
        unsigned int bar = vecWidth / 2;
        while(bar > 0)
        {
        if (id < bar)
        partialSums[t] = partialSums[t] + partialSums[t+bar];
        
        //barrier(CLK_LOCAL_MEM_FENCE);
        //__syncthreads();
        LOCAL_BARRIER;
        bar = bar / 2;
        }            
        // Write result 
        if (id == 0)
        {
        out[myRow]=partialSums[t]; 
        }
        }
        };    // End of cCSR_spmv_vector
        
        
        KERNEL void pELL_spmv_mCoil(    
        const     unsigned int   Reps,            // Number of coils
        const    unsigned int    nRow,        // number of rows
        const    unsigned int    prodJd,     // product of Jd
        const    unsigned int    sumJd,     // sum of Jd
        const    unsigned int    dim,           // dimensionality
        GLOBAL_MEM const unsigned int *Jd,            // Jd, length = dim
        //GLOBAL_MEM const unsigned int *curr_sumJd,            // summation of Jd[0:dimid] 
        GLOBAL_MEM const unsigned int *meshindex,            // meshindex, prodJd * dim
        GLOBAL_MEM const unsigned int *kindx,    // unmixed column indexes of all dimensions
        GLOBAL_MEM const float2 *udata,// interpolation data before Kronecker product
        GLOBAL_MEM const float2 *vec,     // multi-channel kspace data, prodKd * Reps
        GLOBAL_MEM float2 *out)   // multi-channel output, nRow * Reps
        {   
        const unsigned int t = get_local_id(0);
        const unsigned int vecWidth=${LL};
        // Thread ID within wavefront
        const unsigned int id = t & (vecWidth-1);
        
        // One row per wavefront
        unsigned int vecsPerBlock=get_local_size(0)/vecWidth;
        unsigned int myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth); // the myRow-th non-Cartesian sample
        unsigned int m = myRow / Reps;
        unsigned int nc = myRow - m * Reps;
        LOCAL_MEM float2 partialSums[${LL}];
        float2 zero;
        zero.x = 0.0;
        zero.y = 0.0;
        partialSums[t] = zero;
        
        if (myRow < nRow * Reps)
        {
        const unsigned int vecStart = 0; 
        const unsigned int vecEnd =prodJd;             
        float2  y;//=zero;
        
        for (unsigned int j = vecStart+id;  j<vecEnd; j += vecWidth)
        {    // now doing the first dimension
        unsigned int J = Jd[0];
        unsigned int index_shift = m * sumJd ;
        unsigned int index =    index_shift +  meshindex[dim*j + 0];
        unsigned int col = kindx[index] ;
        float2 spdata = udata[index];
        
        index_shift += J; 
        
        for (unsigned int dimid = 1; dimid < dim; dimid ++ )
        {
        unsigned int J = Jd[dimid];
        unsigned int index =  index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
        col += kindx[index] ;//+ 1;                                            // the column index of the current j
        float tmp_x= spdata.x;
        float2 tmp_udata = udata[index];
        spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
        spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
        index_shift  += J;
        }
        float2 vecdata=vec[col * Reps + nc];
        y.x =  spdata.x*vecdata.x - spdata.y*vecdata.y;
        y.y =  spdata.y*vecdata.x + spdata.x*vecdata.y;
        partialSums[t] = y + partialSums[t];
        
        }
        
        LOCAL_BARRIER; 
        
        // Reduce partial sums
        unsigned int bar = vecWidth / 2;
        while(bar > 0)
        {
        if (id < bar)
        partialSums[t]= partialSums[t]+partialSums[t+bar];
        
        //barrier(CLK_LOCAL_MEM_FENCE);
        //__syncthreads();
        LOCAL_BARRIER;
        bar = bar / 2;
        }            
        // Write result 
        if (id == 0)
        {
        out[myRow]=partialSums[t]; 
        }
        }
        };  // End of pELL_spmv_mCoil
        """
    return R

def cMultiplyConjVecInplace():
    """
    Return the kernel source of cMultiplyConjVecInplace
    """
    
    R="""
        KERNEL void cMultiplyConjVecInplace(
        const unsigned int batch, 
        GLOBAL_MEM float2 *a,
        GLOBAL_MEM float2 *outb)
        {
        const unsigned int gid = get_global_id(0);
        const unsigned int voxel_id = (float)gid / (float)batch;
        float2 mul = a[voxel_id]; //  taking the conjugate
        float2 orig = outb[gid];
        float2 tmp;
        tmp.x = orig.x * mul.x + orig.y * mul.y;
        tmp.y = - orig.x * mul.y + orig.y * mul.x; 
        outb[gid] = tmp;
        };
        """
    return R
def cMultiplyConjVec():
    """
    Return the kernel source of cMultiplyConjVec.
    """
    R="""
        KERNEL void cMultiplyConjVec( 
        GLOBAL_MEM float2 *a,
        GLOBAL_MEM float2 *b,
        GLOBAL_MEM float2 *dest)
        {// dest[i]=conj(a[i]) * b[i] 
        const unsigned int i=get_global_id(0);
        dest[i].x=a[i].x*b[i].x+a[i].y*b[i].y;
        dest[i].y=a[i].x*b[i].y-a[i].y*b[i].x;
        };
        """    
    return R
def cMultiplyRealInplace():
    """
    Return the kernel source of cMultiplyRealInplace. 
    """
    R="""
    KERNEL void cMultiplyRealInplace( 
    const unsigned int batch, 
    GLOBAL_MEM const float *a,
    GLOBAL_MEM float2 *outb)
    {
    const unsigned int gid = get_global_id(0);
    // const unsigned int voxel_id = gid / batch;
    const unsigned int voxel_id = (float)gid / (float)batch;
    float mul = a[voxel_id];
    float2 orig = outb[gid];
    orig.x=orig.x*mul; 
    orig.y=orig.y*mul;  
    outb[gid]=orig;
    };
    """
    return R

def cMultiplyVecInplace():
    """
    Return the kernel source of cMultiplyVecInplace.
    """
    R="""
        KERNEL void cMultiplyVecInplace( 
                const unsigned int batch, 
                GLOBAL_MEM const float2 *a,
                GLOBAL_MEM float2 *outb)
        {
        const unsigned int gid = get_global_id(0);
        // const unsigned int voxel_id = gid / batch;
        const unsigned int voxel_id = (float)gid / (float)batch;
        
        float2 mul = a[voxel_id];
        float2 orig = outb[gid];
        float2 tmp;
        tmp.x=orig.x*mul.x-orig.y*mul.y;
        tmp.y=orig.x*mul.y+orig.y*mul.x; 
        outb[gid]=tmp;
        };
        """
    return R
def cAddVec():
    """
    Return the kernel source for cAddVec. 
    """
    R="""
        KERNEL void cAddVec( 
        GLOBAL_MEM float2 *a,
        GLOBAL_MEM float2 *b,
        GLOBAL_MEM float2 *dest)
        {const int i = get_global_id(0);
        dest[i]= a[i]+b[i];
        };"""
    return R

def cDiff():
    """
    Return the kernel source of cDiff.
    """
    R="""
        KERNEL void cDiff(  GLOBAL_MEM    const    int            *order2,
                                               GLOBAL_MEM     const   float2     *indata,
                                               GLOBAL_MEM                  float2     *outdata)
        {
        const unsigned int gid =  get_global_id(0); 
        const unsigned int ind = order2[gid];
        outdata[gid]=indata[ind]- indata[gid];
        };
        """
    return R

def cRealShrink():
    """
    Return the kernel source of xAnisoShrink
    """
    R="""
    KERNEL void xRealShrink(const  float2 threshold,
                                    GLOBAL_MEM const float2 *indata,
                                    GLOBAL_MEM float2 *outdata)
    {
    const unsigned int gid =  get_global_id(0); 
    float2 tmp; // temporay register
    tmp = indata[gid];
    //float zero = 0.0;
    //tmp.x=sign(tmp.x)*max(fabs(tmp.x)-threshold.x, zero); 
    //tmp.y=sign(tmp.y)*max(fabs(tmp.y)-threshold.y, zero); 
    tmp.x =  (tmp.x > threshold.x)*(tmp.x - threshold.x) ; //+ (tmp.x <= - threshold.x)*(tmp.x + threshold.x);
    tmp.y =  (tmp.y > threshold.x)*(tmp.y - threshold.x) ; //+ (tmp.y <= - threshold.x)*(tmp.y + threshold.x);
    outdata[gid]=tmp;
    };
    """
    return R
def cAnisoShrink():
    """
    Return the kernel source of cAnisoShrink
    """
    R="""
    KERNEL void cAnisoShrink(const  float2 threshold,
                                    GLOBAL_MEM const float2 *indata,
                                    GLOBAL_MEM  float2 *outdata)
    {
    const unsigned int gid =  get_global_id(0); 
    float2 tmp; // temporay register
    tmp = indata[gid];
    //float zero = 0.0;
    //tmp.x=sign(tmp.x)*max(fabs(tmp.x)-threshold.x, zero); 
    //tmp.y=sign(tmp.y)*max(fabs(tmp.y)-threshold.y, zero); 
    tmp.x =  (tmp.x > threshold.x)*(tmp.x - threshold.x) + (tmp.x < - threshold.x)*(tmp.x + threshold.x);
    tmp.y =  (tmp.y > threshold.x)*(tmp.y - threshold.x) + (tmp.y < - threshold.x)*(tmp.y + threshold.x);
    outdata[gid]=tmp;
    };
    """
    return R

def cSqrt():
    """
    Return the kernel source of cSqrt.
    """
    
    R="""
        KERNEL void cSqrt( 
        GLOBAL_MEM  float2 *CX)
        {
        // Copy x to y: y = x;
        //CX: input output array (float2)
        
        int gid=get_global_id(0);  
        CX[gid].x=sqrt(CX[gid].x);
        };
        """  
    return R

def cSelect():
    """
    Return the kernel source of cSelect. 
    """
    R="""
        KERNEL void cSelect(
        GLOBAL_MEM const  int *order1,
        GLOBAL_MEM const  int *order2,
        GLOBAL_MEM const float2 *indata,
        GLOBAL_MEM       float2 *outdata)
        {
        const unsigned int gid=get_global_id(0); 
        outdata[order2[gid]]=
                   indata[order1[gid]];
        };
        """
    return R