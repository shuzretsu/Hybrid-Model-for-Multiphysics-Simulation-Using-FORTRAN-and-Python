import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def visualize_heat(T, n):
    #Visualizes the heat distribution.
    plt.imshow(T, cmap=cm.hot, interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title('Heat Distribution')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def visualize_fluid(U, V, n):
    #Visualizes the velocity field of fluid flow.
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    plt.figure()
    plt.quiver(X, Y, U, V)
    plt.title('Fluid Flow Velocity Field')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def visualize_molecular_positions(positions):
    #Visualizes molecular dynamics positions in 2D.
    plt.figure()
    plt.scatter(positions[:, 0], positions[:, 1], s=5)
    plt.title('Molecular Dynamics - Particle Positions')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


if __name__ == "__main__":
    n = 50
    
    # Generate some example data for visualization:
    T = np.random.rand(n, n) * 100  # Random heat distribution
    U = np.random.rand(n, n)  # Random velocity field (x-component)
    V = np.random.rand(n, n)  # Random velocity field (y-component)
    
    # Molecular dynamics particle positions
    positions = np.random.rand(100, 2) * n  # Random particle positions
    
    # Visualize heat distribution
    visualize_heat(T, n)
    
    # Visualize fluid flow velocity field
    visualize_fluid(U, V, n)
    
    # Visualize molecular positions
    visualize_molecular_positions(positions)
