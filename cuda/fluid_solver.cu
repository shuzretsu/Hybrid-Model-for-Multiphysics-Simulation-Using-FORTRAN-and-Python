__global__ void fluid_kernel(float *U, float *V, float *P, int n, float dt, float dx, float dy, float rho, float mu) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n*n) return;
    int i = idx / n;
    int j = idx % n;

    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        U[idx] = U[idx] + dt * (-U[idx] * (U[(i+1)*n+j] - U[(i-1)*n+j]) / (2*dx) +
                                 mu * (U[(i+1)*n+j] - 2*U[idx] + U[(i-1)*n+j]) / (dx*dx));
        V[idx] = V[idx] + dt * (-V[idx] * (V[i*n+(j+1)] - V[i*n+(j-1)]) / (2*dy) +
                                 mu * (V[i*n+(j+1)] - 2*V[idx] + V[i*n+(j-1)]) / (dy*dy));
        P[idx] = P[idx] + dt * (-rho * ((U[(i+1)*n+j] - U[(i-1)*n+j]) / (2*dx) +
                                        (V[i*n+(j+1)] - V[i*n+(j-1)]) / (2*dy)));
    }
}
