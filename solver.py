import json
from math import *
import matplotlib.pyplot as plt
import csv
from sympy import *
import numpy as np
from numpy.linalg import inv

# Неявная схема

# загрузка данных
with open("task.json", 'r') as task_file:
    task = json.load(task_file)

# определение параметров сетки
h = task["solver"]["h"]
tau = task["solver"]["tau"]
MESH_X_SIZE = int((task["x1"]-task["x0"]) / h)
MESH_T_SIZE = int(1 / tau)

# функция, задающая начальные условия
expr1 = parse_expr(task["ICS"])
x = Symbol("x")
ICS = lambdify(x, expr1)

# функция, задающая правую часть
expr2 = parse_expr(task["rightPart"])
t = Symbol("t")
f = lambdify([t,x], expr2)

# функции, задающие граничные условия в зависимости от их вида
# номер типа граничного условия справа и слева (1го или 2го рода)
BC_0_num = task["BC_num"][0]
BC_1_num = task["BC_num"][1]

BC_name = "" # название массива, откуда будет взята функция, в зависимости от вида граничного условия
BC_scale = 1 # масштабный параметр для граничного условия - если это производная, то значение функции, взятой из файла, домножается еще на величину шага сетки

if BC_0_num == 1:
    BC_name = "BC_1"
elif BC_0_num == 2:
    BC_name = "BC_2"
    BC_scale = 2*h
# определение функции - ГУ слева
beta0_buf = lambdify(t, task[BC_name][0])
beta0 = lambda t: beta0_buf(t)*BC_scale

if BC_1_num == 1:
    BC_name = "BC_1"
    BC_scale = 1
elif BC_1_num == 2:
    BC_name = "BC_2"
    BC_scale = 2*h
# определение функции - ГУ справа
beta1_buf = lambdify(t, task[BC_name][1])
beta1 = lambda t: beta1_buf(t)*BC_scale

# определение матрицы СЛАУ
a = task["a"]
p = -(a ** 2) * tau / (h ** 2)
q = 1 + 2 * (a ** 2) * tau / (h ** 2)
r = p
H = np.zeros((MESH_X_SIZE + 1, MESH_X_SIZE + 1))

# трех-диагональный участок, одинаковый для всех ГУ
for i in range(1, H.shape[0] - 1):
    H[i, i - 1] = p
    H[i, i] = q
    H[i, i + 1] = r

# учет ГУ в СЛАУ:
if BC_0_num == 1:
    H[0, 0] = 1
elif BC_0_num == 2:
    H[0, 0] = 3
    H[0, 1] = -4
    H[0, 2] = 1
if BC_1_num == 1:
    H[MESH_X_SIZE, MESH_X_SIZE] = 1
elif BC_1_num == 2:
    H[MESH_X_SIZE, MESH_X_SIZE] = 3
    H[MESH_X_SIZE, MESH_X_SIZE-1] = -4
    H[MESH_X_SIZE, MESH_X_SIZE - 2] = 1

U = np.zeros((MESH_T_SIZE + 1, MESH_X_SIZE + 1))

# начальные условия
U[0] = np.array([ICS(k * h) for k in range(MESH_X_SIZE + 1)])

# задание начального сечения температуры
U[0] = np.array([ICS(k * h) for k in range(MESH_X_SIZE + 1)])
t1 = 0
# Решение СЛАУ
for j in range(1, MESH_T_SIZE + 1):
    F = np.zeros((MESH_X_SIZE + 1, 1))
    t1 += tau
    x1 = 0
    # формирование вектора-правой части в СЛАУ
    for i in range(1, MESH_X_SIZE):
        F[i, 0] = tau * f(t1, x1)+U[j - 1][i]
        x1 += h
    # учет граничных значений в  векторе-правой части
    F[0, 0] = beta0(t1)
    F[MESH_X_SIZE, 0] = beta1(t1)
    U[j] = (np.dot(inv(H), F)).reshape((MESH_X_SIZE + 1))

# печать результата
u = U[1]
plt.plot(np.linspace(0, 1, MESH_X_SIZE + 1), u)
plt.show()
