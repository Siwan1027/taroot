import math
import numpy as np


# 거리 계산 func
def leng_cal(sp,tp):
    # 가로,세로 길이 단위
    x_length = int(round(abs(sp[0] - tp[0])*1000000))
    y_length = int(round(abs(sp[1] - tp[1])*1000000))
    # 직선거리 구하기 공식 | m 단위로 올림
    shortest_route = math.sqrt(x_length**2 + y_length**2)
    # 넓이 구하기 제곱키로미터
    # width = abs((sp[0] - tp[0])) * abs((sp[1] - tp[1]))
    return round(shortest_route)


# 좌표평면 생성 func
def make_matrix(sp, tp):
    # 직선거리 산출
    shortest_route = leng_cal(sp, tp)

    # 직선거리 중간지점
    center = [(sp[0] + tp[0]) / 2, (sp[1] + tp[1]) / 2]

    # 거리로 좌표평면 상 좌표별 거리 설정
    if shortest_route <= 14141:  # 10km 이하
        i = 0.0001
    else:  # 10키로 미터 초과
        i = 0.001

    # 좌표평면 별 구간 설정 및 좌표 생성
    n = 4 if shortest_route <= 14141 else 3
    q_1_x, q_1_y = np.meshgrid(np.arange(center[0], max(sp[0], tp[0]) + i, i), np.arange(center[1], max(sp[1], tp[1]) + i, i))
    q_1 = [[round(x, n), round(y, n)] for x, y in zip(q_1_x.ravel(), q_1_y.ravel())]

    q_2_x, q_2_y = np.meshgrid(np.arange(center[0], min(sp[0], tp[0]), -i), np.arange(center[1], max(sp[1], tp[1]) + i, i))
    q_2 = [[round(x, n), round(y, n)] for x, y in zip(q_2_x.ravel(), q_2_y.ravel())]

    q_3_x, q_3_y = np.meshgrid(np.arange(center[0], min(sp[0], tp[0]), -i), np.arange(center[1], min(sp[1], tp[1]), -i))
    q_3 = [[round(x, n), round(y, n)] for x, y in zip(q_3_x.ravel(), q_3_y.ravel())]

    q_4_x, q_4_y = np.meshgrid(np.arange(center[0], max(sp[0], tp[0]) + i, i), np.arange(center[1], min(sp[1], tp[1]), -i))
    q_4 = [[round(x, n), round(y, n)] for x, y in zip(q_4_x.ravel(), q_4_y.ravel())]

    result = {'Quadrant 1': q_1, 'Quadrant 2': q_2, 'Quadrant 3': q_3, 'Quadrant 4': q_4}

    return result, center