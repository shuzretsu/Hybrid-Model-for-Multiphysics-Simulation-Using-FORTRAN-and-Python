import numpy as np
import matplotlib.pyplot as plt
from heat_solver import heat_conduction
from fluid_solver import fluid_flow
from cuda_acceleration import run_cuda_heat_conduction

# User input
n = int(input("Enter grid size (e.g., 50): "))
dx = float(input("Enter spatial step size (e.g., 0.01): "))
dy = dx
dt = float(input("Enter time step (e.g., 0.0001): "))
kappa = float(input("Enter thermal diffusivity (e.g., 0.01): "))
rho = float(input("Enter fluid density (e.g., 1.0): "))
mu = float(input("Enter fluid viscosity (e.g., 0.01): "))

# Initialize fields
T = np.zeros((n, n))
initial_temp = float(input("Enter initial temperature (e.g., 100): "))
T[20:30, 20:30] = initial_temp

U = np.zeros((n, n))
V = np.zeros((n, n))
P = np.zeros((n, n))

# Run heat conduction
run_cuda_heat_conduction(T, n, dt, dx, kappa)

# Run fluid flow
fluid_flow(U, V, P, n, dt, dx, dy, rho, mu)

# Save and visualize results
np.savetxt('data/results/temperature.txt', T)
np.savetxt('data/results/velocity.txt', U)

plt.imshow(T, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.savefig('data/results/heat_map_output.png')
plt.show()

plt.quiver(U, V)
plt.savefig('data/results/velocity_field_output.png')
plt.show()
