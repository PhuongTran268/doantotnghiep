import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from docplex.mp.model import Model
from cplex import Cplex

# Nhập dữ liệu
n = 6  # Đơn hàng
m = 5    #máy
hj = [2, 2, 2, 2, 2, 2]
ki = [2, 2, 2, 2, 4]
# Khả năng gia công
a = [
    # Máy 1
    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    # máy 2
    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    # máy 3
    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    # máy 4
    [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
    # máy 5
    [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
]

# Thời gian xử lý
pi = [
    # Máy 1
    [[19.5, 0], [27.5, 0], [14.16666, 0], [11, 0], [20.08933, 0], [15.93, 0]],
    # máy 2
    [[9.75, 0], [13.75, 0], [17, 0], [13.2, 0], [24.1072, 0], [19.12, 0]],
    # máy 3
    [[19.02439, 0], [26.82927, 0], [20.7317, 0], [16.09756, 0], [29.399, 0], [23.317, 0]],
    # máy 4
    [[0, 10.725], [0, 15.125], [0, 11.6875], [0, 9.075], [0, 16.5737], [0, 13.145]],
    # máy 5
    [[0, 7.722], [0, 10.89], [0, 8.415], [0, 6.534], [0, 11.93306], [0, 9.4644]]
]

# Khả năng gia công
a = [
    # Máy 1
    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    # máy 2
    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    # máy 3
    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    # máy 4
    [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
    # máy 5
    [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
]

# Thời gian xử lý
pi = [
    # Máy 1
    [[19.5, 0], [27.5, 0], [14.16666, 0], [11, 0], [20.08933, 0], [15.93, 0]],
    # máy 2
    [[9.75, 0], [13.75, 0], [17, 0], [13.2, 0], [24.1072, 0], [19.12, 0]],
    # máy 3
    [[19.02439, 0], [26.82927, 0], [20.7317, 0], [16.09756, 0], [29.399, 0], [23.317, 0]],
    # máy 4
    [[0, 10.725], [0, 15.125], [0, 11.6875], [0, 9.075], [0, 16.5737], [0, 13.145]],
    # máy 5
    [[0, 7.722], [0, 10.89], [0, 8.415], [0, 6.534], [0, 11.93306], [0, 9.4644]]
]

dj = [24, 24, 24, 24, 48, 48]
wj = [5, 4, 2, 5, 4, 3]
L = 1000

# Khởi tạo mô hình
model = Model(name='Production Scheduling')

# Biến quyết định
y = {(i, j, h): model.binary_var(name=f'y_{i}_{j}_{h}') for i in range(m) for j in range(n) for h in range(hj[j])}
x = {(i, j, h, k): model.binary_var(name=f'x_{i}_{j}_{h}_{k}') for i in range(m) for j in range(n) for h in range(hj[j]) for k in range(ki[i])}
t = {(j, h): model.continuous_var(name=f't_{j}_{h}') for j in range(n) for h in range(hj[j])}
T = {(i, k): model.continuous_var(name=f'T_{i}_{k}') for i in range(m) for k in range(ki[i])}
P = {(j, h): model.continuous_var(name=f'P_{j}_{h}') for j in range(n) for h in range(hj[j])}
Cj = {j: model.continuous_var(name=f'Cj_{j}') for j in range(n)}
Tj = {j: model.continuous_var(name=f'Tj_{j}') for j in range(n)}

# Ràng buộc 7: y[i,j,h] <= a[i,j,h] với i = 1, …, m; j = 1, …, n; h = 1, …, hj
for i in range(m):
    for j in range(n):
        for h in range(hj[j]):
            model.add_constraint(
                ct=y[i, j, h] <= a[i][j][h],
                ctname=f'c10_{i}_{j}_{h}'
            )

# Ràng buộc 8: Mỗi đơn hàng chỉ được gán cho một máy tại một thứ tự nhất định
for i in range(m):
    for k in range(ki[i]):
        model.add_constraint(
            ct=sum(x[i, j, h, k] for j in range(n) for h in range(hj[j])) == 1,
            ctname=f'c11_{i}_{k}'
        )

# Ràng buộc 9: Mỗi công đoạn của mỗi đơn hàng phải được thực hiện trên một máy duy nhất
for j in range(n):
    for h in range(hj[j]):
        model.add_constraint(
            ct=sum(y[i, j, h] for i in range(m)) == 1,
            ctname=f'c12_{j}_{h}'
        )

# Ràng buộc 10: Chọn thứ tự gia công k trên máy i cho đơn hàng
for i in range(m):
    for j in range(n):
        for h in range(hj[j]):
            model.add_constraint(
                ct=sum(x[i, j, h, k] for k in range(ki[i])) == y[i, j, h],
                ctname=f'c13_{i}_{j}_{h}'
            )

# Ràng buộc 2: Tính P
for j in range(n):
    for h in range(hj[j]):
        lin_expr = [y[i, j, h] * pi[i][j][h] for i in range(m)]
        model.add_constraint(
            ct=model.sum(lin_expr) == P[j, h],
            ctname=f'constraint_P_{j}_{h}'
        )

# Ràng buộc số 3: t[j,h] + P[j,h] <= t[j,h+1] với j = 1, …, n; h = 1, …, hj-1
for j in range(n):
    for h in range(max(hj) - 1):
        model.add_constraint(
            ct=t[j, h] + P[j, h] <= t[j, h + 1],
            ctname=f'c6_{j}_{h}'
        )

# Ràng buộc 4: T[i,k] + P[j,h]*x[i,j,h,k] <= T[i,k+1] với i = 1, …, m; j = 1, …, n; h = 1, …, hj; k = 1, …, ki-1
for i in range(m):
    for k in range(ki[i] - 1):
        for j in range(n):
            for h in range(hj[j]):
                model.add_constraint(
                    ct=T[i, k] + P[j, h] * x[i, j, h, k] <= T[i, k + 1],
                    ctname=f'c7_{i}_{k}_{j}_{h}'
                )

#In ra các ràng buộc của mô hình
for ct in model.iter_constraints():
    print(ct)
    
# Ràng buộc 5: T[i,k] <= t[j,h] + (1-x[i,j,h,k])*L với i = 1, …, m; j = 1, …, n; h = 1, …, hj; k = 1, …, ki
for i in range(m):
    for k in range(ki[i]):
        for j in range(n):
            for h in range(hj[j]):
                model.add_constraint(
                    ct=T[i, k] <= t[j, h] + (1 - x[i, j, h, k]) * L,
                    ctname=f'c8_{i}_{k}_{j}_{h}'
                )

# Ràng buộc 6: T[i,k] + (1-x[i,j,h,k])*L >= t[j,h] với i = 1, …, m; j = 1, …, n; h = 1, …, hj; k = 1, …, ki
for i in range(m):
    for k in range(ki[i]):
        for j in range(n):
            for h in range(hj[j]):
                model.add_constraint(
                    ct=T[i, k] + (1 - x[i, j, h, k]) * L >= t[j, h],
                    ctname=f'c9_{i}_{k}_{j}_{h}'
                )


# Ràng buộc 11: Thời gian hoàn thành của mỗi đơn hàng bằng thời gian bắt đầu gia công công đoạn cuối cùng cộng với thời gian xử lý công đoạn cuối cùng
for j in range(n):
    model.add_constraint(
        ct=Cj[j] == t[j, hj[j] - 1] + P[j, hj[j] - 1],
        ctname=f'c14_{j}'
    )

# Ràng buộc 12: Tính thời gian trễ của mỗi đơn hàng
for j in range(n):
    model.add_constraint(
        ct=Tj[j] >= Cj[j] - dj[j],
        ctname=f'c15_{j}_lowerbound'
    )
    model.add_constraint(
        ct=Tj[j] >= 0,
        ctname=f'c15_{j}_nonnegative'
    )

# Ràng buộc 13: Biến quyết định x chỉ nhận giá trị 0 hoặc 1
for i in range(m):
    for j in range(n):
        for h in range(hj[j]):
            for k in range(ki[i]):
                model.add_constraint(
                    x[i, j, h, k] <= 1,
                    ctname=f'c16_{i}_{j}_{h}_{k}_upperbound'
                )
                model.add_constraint(
                    x[i, j, h, k] >= 0,
                    ctname=f'c16_{i}_{j}_{h}_{k}_lowerbound'
                )

# Ràng buộc 14: Biến quyết định y chỉ nhận giá trị 0 hoặc 1
for i in range(m):
    for j in range(n):
        for h in range(hj[j]):
            model.add_constraint(
                y[i, j, h] <= 1,
                ctname=f'c17_{i}_{j}_{h}_upperbound'
            )
            model.add_constraint(
                y[i, j, h] >= 0,
                ctname=f'c17_{i}_{j}_{h}_lowerbound'
            )


# Ràng buộc 15: Đảm bảo các biến liên quan đến thời gian là số dương
for j in range(n):
    for h in range(hj[j]):
        model.add_constraint(
            t[j, h] >= 0,
            ctname=f'c15_t_{j}_{h}'
        )
for j in range(n):
    model.add_constraint(
        Tj[j] >= 0,
        ctname=f'c15_T_{j}'
    )

for i in range(m):
    for k in range(ki[i]):
        model.add_constraint(
            T[i, k] >= 0,
            ctname=f'c15_T_{i}_{k}'
        )

# Hàm mục tiêu
obj_expr = model.sum(wj[j] * Tj[j] for j in range(n))
model.minimize(obj_expr)

# Giải bài toán tối ưu
model.solve()

# In ra giá trị của các biến sau khi giải quyết mô hình
print("Các giá trị của các biến sau khi giải quyết mô hình:")
for var in model.iter_variables():
    print(var, var.solution_value)

print("Objective value:", model.objective_value)



