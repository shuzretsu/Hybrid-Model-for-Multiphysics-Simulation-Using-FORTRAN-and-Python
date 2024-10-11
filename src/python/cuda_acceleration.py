import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void heat_kernel(float *T, int n, float dt, float dx, float kappa) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n*n) return;
    int i = idx / n;
    int j = idx % n;
    
    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        T[idx] = T[idx] + kappa * dt * ((T[(i+1)*n+j] - 2*T[idx] + T[(i-1)*n+j]) / (dx*dx) +
                                         (T[i*n+(j+1)] - 2*T[idx] + T[i*n+(j-1)]) / (dx*dx));
    }
}
""")

heat_kernel = mod.get_function("heat_kernel")

def run_cuda_heat_conduction(T, n, dt, dx, kappa):
    T_gpu = drv.mem_alloc(T.nbytes)
    drv.memcpy_htod(T_gpu, T)
    heat_kernel(T_gpu, np.int32(n), np.float32(dt), np.float32(dx), np.float32(kappa),
                block=(256,1,1), grid=(int((n*n+255)/256),1))
    drv.memcpy_dtoh(T, T_gpu)
