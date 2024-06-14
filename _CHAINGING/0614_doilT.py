import numpy as np

def find_intersection_points(a, x, y, z, r):
    A = r**2 - 2 * a * x + a**2
    B = 2 * a * (x - a)
    C = a**2 - r**2
    
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return None  # No real roots, no intersection
    
    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)
    
    points = []
    for t in [t1, t2]:
        x_prime = a + t * (x - a)
        y_prime = t * y
        z_prime = -t * z
        points.append((x_prime, y_prime, z_prime))
    
    return points

# Example values
a = 2
x = 1
y = 2
z = -3
r = 5

intersection_points = find_intersection_points(a, x, y, z, r)
print(intersection_points)
