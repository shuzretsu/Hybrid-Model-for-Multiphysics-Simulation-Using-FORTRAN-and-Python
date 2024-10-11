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
