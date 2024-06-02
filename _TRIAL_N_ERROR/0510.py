import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R = 32  # 구의 반경
window_width = 8  # 창문의 너비
window_height = 16  # 창문의 높이
dx = 3  # 이동 거리

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 구 그리기
phi, theta = np.mgrid[0.0:2.0*np.pi:40j, 0.0:np.pi:20j]
x = R * np.sin(theta) * np.cos(phi)
y = R * np.sin(theta) * np.sin(phi)
z = R * np.cos(theta)
ax.plot_wireframe(x, y, z, color='black', linewidth=0.25)

# 사람 그리기
ax.plot([0], [0], [0], 'ro', markersize=5)
ax.plot([dx], [0], [0], 'ro', markersize=5)

# 창문 그리기
wx = np.array([window_width/2, -window_width/2, -window_width/2, window_width/2])
wy = np.array([window_height/2, window_height/2, -window_height/2, -window_height/2])
wz = np.zeros_like(wx)
ax.plot(wx, wy, wz, 'g-')

# 직선 그리기 및 교점 계산
for x_start in [0, dx]:
    for point in zip(wx, wy, wz):
        direction = np.array(point) - np.array([x_start, 0, 0])
        norm_direction = direction / np.linalg.norm(direction)
        
        # 광선-구 충돌 감지
        a = np.dot(norm_direction, norm_direction)
        b = 2 * np.dot(norm_direction, np.array([x_start, 0, 0]))
        c = np.dot([x_start, 0, 0], [x_start, 0, 0]) - R**2
        
        discriminant = b**2 - 4 * a * c
        if discriminant >= 0:
            t1 = (-b - np.sqrt(discriminant)) / (2 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2 * a)
            t = max(t1, t2)  # 구의 표면 외부에서 시작하므로 큰 값을 선택
            intersection_point = np.array([x_start, 0, 0]) + t * norm_direction
            ax.plot([x_start, intersection_point[0]], [0, intersection_point[1]], [0, intersection_point[2]], 'b-', linewidth=2)

# 그리드 및 레이블 설정
ax.set_xlim([-R, R])
ax.set_ylim([-R, R])
ax.set_zlim([-R, R])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
