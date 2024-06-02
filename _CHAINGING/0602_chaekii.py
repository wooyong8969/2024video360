'''/**dx, dy 지점에서 창문 너머 시야 계산 class로 객체화**/''' 
import numpy as np

class chaekii:
    def __init__(self, R, box_size, dx, dy):
        self.R = R
        self.half_size = box_size / 2
        self.dx = dx
        self.dy = dy
        self.positions = [
            np.array([0, 0, 0]),
            np.array([dx, 0, 0]),
            np.array([0, dy, 0]),
            np.array([dx, dy, 0])
        ]
        self.window_corners = np.array([
            [self.half_size, self.half_size, self.half_size],
            [-self.half_size, self.half_size, self.half_size],
            [self.half_size, self.half_size, -self.half_size],
            [-self.half_size, self.half_size, -self.half_size]
        ])
    
    def calculate_intersections(self):
        intersection_points = []
        for position in self.positions:
            for corner in self.window_corners:
                direction = corner - position
                norm_direction = direction / np.linalg.norm(direction)
                
                a = np.dot(norm_direction, norm_direction)
                b = 2 * np.dot(norm_direction, position)
                c = np.dot(position, position) - self.R**2

                discriminant = b**2 - 4 * a * c
                if discriminant >= 0:
                    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                    t2 = (-b + np.sqrt(discriminant)) / (2 * a)
                    if t1 > 0 and t2 > 0:
                        t = min(t1, t2)
                    elif t1 > 0:
                        t = t1
                    elif t2 > 0:
                        t = t2
                    else:
                        continue
                    intersection_point = position + t * norm_direction
                    intersection_points.append(intersection_point)
        return intersection_points


calculator = chaekii(R=300, box_size=200, dx=75, dy=60)
intersection_points = calculator.calculate_intersections()

# 교점 좌표 출력 (직교 좌표계)
print("Intersection Points (Cartesian Coordinates):")
for i, point in enumerate(intersection_points):
    print(f"Intersection {i}: {point}")
