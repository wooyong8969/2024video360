import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_view_angle(position, corner):
    vector = corner - position
    return vector / np.linalg.norm(vector)

R = 300
box_size = 200
half_size = box_size / 2
dx = 75
dy = 60

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

corner = np.array([half_size, half_size, half_size])

# 사람 위치 정의
positions = [
    np.array([0, 0, 0]),
    np.array([dx, 0, 0]),
    np.array([0, dy, 0]),
    np.array([dx, dy, 0])
]

for position in positions:
    ax.plot([position[0]], [position[1]], [position[2]], 'o', markersize=5)

# 벡터 그리기 및 각도 계산
previous_vector = calculate_view_angle(positions[0], corner)
angles = []
for i, position in enumerate(positions[1:], 1):
    current_vector = calculate_view_angle(position, corner)
    cos_theta = np.dot(previous_vector, current_vector)
    angle = np.degrees(np.arccos(cos_theta))
    angles.append(angle)
    
    # 벡터 그리기
    ax.plot([position[0], corner[0]], [position[1], corner[1]], [position[2], corner[2]], linewidth=2)
    previous_vector = current_vector

# 창문 모퉁이 그리기
ax.plot([corner[0]], [corner[1]], [corner[2]], 's', color='red', markersize=8)

# 축 설정
ax.set_xlim([-R, R])
ax.set_ylim([-R, R])
ax.set_zlim([-R, R])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# 각도 변화 출력
for idx, angle in enumerate(angles, 1):
    print(f"Position {idx-1} to {idx}: Angle change = {angle:.2f} degrees")
