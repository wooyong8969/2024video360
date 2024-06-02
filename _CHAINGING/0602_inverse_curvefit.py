'''/**dy와 변화 시야각 theta 사이의 관계식 파라미터 계산 - 실패**/''' 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 역수 관계
def inverse_relation(dy, k, c):
    return k / (dy + c)

R = 300
box_size = 200
half_size = box_size / 2
dx = 75

corner = np.array([half_size, half_size, half_size])

def calculate_angle(dy):
    position = np.array([dx, dy, 0])
    vector = corner - position
    norm_vector = vector / np.linalg.norm(vector)
    initial_vector = corner / np.linalg.norm(corner)
    cos_theta = np.dot(initial_vector, norm_vector)
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    return angle

# 데이터 생성
dy_values = np.linspace(1, 100, 400)  # dy가 0인 경우를 제외
angles = np.array([calculate_angle(dy) for dy in dy_values])

# 데이터 시각화
plt.figure(figsize=(10, 5))
plt.plot(dy_values, angles, label='Angle vs. dy')
plt.xlabel('dy (shift in y-axis)')
plt.ylabel('Angle change (degrees)')
plt.title('Angle Change vs. dy (Inverse Relation)')
plt.legend()
plt.grid(True)
plt.show()

# 데이터에 모델 피팅
params, params_covariance = curve_fit(inverse_relation, dy_values, angles, p0=[100, 10])
k, c = params

print(f"Estimated parameters: k = {k:.4f}, c = {c:.4f}")