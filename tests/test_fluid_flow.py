import numpy as np
from fluid_solver import fluid_flow

def test_fluid_flow():
    n = 10
    U = np.zeros((n, n))
    V = np.zeros((n, n))
    P = np.zeros((n, n))
    dt = 0.001
    dx = 0.1
    dy = 0.1
    rho = 1.0
    mu = 0.01

    fluid_flow(U, V, P, n, dt, dx, dy, rho, mu)

    assert np.all(U >= 0), "Velocity U should be non-negative"
    assert np.all(V >= 0), "Velocity V should be non-negative"

if __name__ == "__main__":
    test_fluid_flow()
    print("Fluid flow test passed.")
