import numpy as np
from heat_solver import heat_conduction

def test_heat_conduction():
    n = 10
    T = np.zeros((n, n))
    kappa = 0.01
    dt = 0.001
    dx = 0.1
    
    heat_conduction(T, n, dt, dx, kappa)
    
    assert np.all(T >= 0), "Temperature values should be non-negative"

if __name__ == "__main__":
    test_heat_conduction()
    print("Heat conduction test passed.")
