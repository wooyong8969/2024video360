'''/**dy 라인에서 dx 만큼 이동 시 변화하는 시야각 계산**/'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_view_angle(position, corner):
    vector = corner - position
    return vector / np.linalg.norm(vector)

R = 30
box_size = 10
half_size = box_size / 2
dx = 5
dy = -70

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 구 그리기
phi, theta = np.mgrid[0.0:2.0*np.pi:40j, 0.0:np.pi:20j]
x = R * np.sin(theta) * np.cos(phi)
y = R * np.sin(theta) * np.sin(phi)
z = R * np.cos(theta)
ax.plot_wireframe(x, y, z, color='black', linewidth=0.25)

# 창문 모서리 정의
window_corners = np.array([
    [half_size, half_size, half_size],
    [-half_size, half_size, half_size],
    [half_size, half_size, -half_size],
    [-half_size, half_size, -half_size]
])

# 사람 위치 정의
positions = [
    np.array([0, 0, 0]),
    np.array([dx, 0, 0]),
    #np.array([0, dy, 0]),
    #np.array([dx, dy, 0])
]

# 각 위치에서 교점 계산 및 시선 표시
colors = ['blue', 'green', 'orange', 'purple']
intersection_points = []  # 교점 좌표 저장 리스트

for position, color in zip(positions, colors):
    ax.plot([position[0]], [position[1]], [position[2]], 'o', color=color, markersize=5)
    for corner in window_corners:
        direction = corner - position
        norm_direction = direction / np.linalg.norm(direction)

        # 교점 구하기
        a = np.dot(norm_direction, norm_direction)
        b = 2 * np.dot(norm_direction, position)
        c = np.dot(position, position) - R**2

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
            intersection_points.append(intersection_point)  # 교점 좌표 저장
            ax.plot([position[0], intersection_point[0]], [position[1], intersection_point[1]], [position[2], intersection_point[2]], color=color, linewidth=2)

# 창문 전체 모서리 그리기
corners = np.array([
    [half_size, half_size, half_size],
    [half_size, half_size, -half_size],
    #[half_size, -half_size, half_size],
    #[half_size, -half_size, -half_size],
    [-half_size, half_size, half_size],
    [-half_size, half_size, -half_size],
    #[-half_size, -half_size, half_size],
    #[-half_size, -half_size, -half_size]
])
for start in corners:
    for end in corners:
        if np.linalg.norm(start-end) == box_size:
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-', linewidth=1)

ax.set_xlim([-R, R])
ax.set_ylim([-R, R])
ax.set_zlim([-R, R])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# 교점 좌표 출력
for i, point in enumerate(intersection_points):
    print(f"Intersection {i}: {point}")

