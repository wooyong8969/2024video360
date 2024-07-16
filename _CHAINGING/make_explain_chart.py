import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def calculate_plane_equation(corners):
    p1, p2, p3, p4 = corners
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    
    A, B, C = normal
    D = -np.dot(normal, p1)
    
    return A, B, C, D

def calculate_corners(center, width, height):
    cx, cy, cz = center
    half_width = width / 2.0
    half_height = height / 2.0

    return [
        (cx - half_width, cz, cy - half_height),  # Bottom left
        (cx + half_width, cz, cy - half_height),  # Bottom right
        (cx + half_width, cz, cy + half_height),  # Top right
        (cx - half_width, cz, cy + half_height)   # Top left
    ]

def plot_scene(center, width, height, radius):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the window
    corners = calculate_corners(center, width, height)
    xs, zs, ys = zip(*corners)
    
    A, B, C, D = calculate_plane_equation(corners)
    X, Z = np.meshgrid(np.linspace(min(xs), max(xs), 10), np.linspace(min(zs), max(zs), 10))
    Y = (-D - A * X - C * Z) / B
    
    ax.plot_surface(X, Z, Y, alpha=0.5, rstride=100, cstride=100, color='b')
    ax.scatter(xs, zs, ys, color='r')
    
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ax.plot([xs[i], xs[j]], [zs[i], zs[j]], [ys[i], ys[j]], color='r')
    
    # Plot the sphere with lightblue color
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.5)

    ax.scatter([0], [0], [0], color='red', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

center = (-50, 50, 50)  # Center of the window
width = 80             # Width of the window
height = 40            # Height of the window
R = 200                # Radius of the sphere
plot_scene(center, width, height, R)
