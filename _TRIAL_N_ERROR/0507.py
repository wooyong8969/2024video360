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
window_corners = np.array([
    [window_width / 2, window_height / 2, 0],
    [-window_width / 2, window_height / 2, 0]
])
wx = np.array([window_width/2, -window_width/2, -window_width/2, window_width/2])
wy = np.array([window_height/2, window_height/2, -window_height/2, -window_height/2])
wz = np.zeros_like(wx)
ax.plot(wx, wy, wz, 'g-')

# 직선 그리기 및 교점 계산
for cusser in [0, dx]:
    x_start = cusser
    for point in window_corners:
        direction = point - np.array([x_start, 0, 0])
        t = np.linspace(0, 2 * R, 400)
        x_line = x_start + t * direction[0] / R
        y_line = t * direction[1] / R
        z_line = t * direction[2] / R

        # 직선과 구의 교점 계산
        intersection_idx = np.where(x_line**2 + y_line**2 + z_line**2 <= R**2)[0]
        if len(intersection_idx) > 0:
            idx = intersection_idx[-1]
            ax.plot([x_start, x_line[idx]], [0, y_line[idx]], [0, z_line[idx]], 'b-', linewidth=2)

# 그리드 및 레이블 설정
ax.set_xlim([-R, R])
ax.set_ylim([-R, R])
ax.set_zlim([-R, R])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()