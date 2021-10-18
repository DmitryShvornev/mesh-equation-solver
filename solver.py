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

# функция, задающая начальные условия
expr1 = parse_expr(task["ICS"])
x = Symbol("x")
ICS = lambdify(x, expr1)

# функция, задающая правую часть
expr2 = parse_expr(task["rightPart"])
t = Symbol("t")
f = lambdify([t,x], expr2)

h = task["solver"]["h"]
tau = task["solver"]["tau"]
a = task["a"]
p = -(a ** 2) * tau / (h ** 2)
q = 1 + 2 * (a ** 2) * tau / (h ** 2)
r = p
MESH_X_SIZE = int((task["x1"]-task["x0"]) / h)
MESH_T_SIZE = int(1 / tau)
beta0 = lambdify(t, task["BC_1"][0])
beta1 = lambdify(t, task["BC_1"][1])

dzeta0 = lambdify(t, task["BC_2"][0])
dzeta1 = lambdify(t, task["BC_2"][1])

H = np.zeros((MESH_X_SIZE + 1, MESH_X_SIZE + 1))
for i in range(1, H.shape[0] - 1):
    H[i, i - 1] = p
    H[i, i] = q
    H[i, i + 1] = r
H[0, 0] = 1
H[MESH_X_SIZE, MESH_X_SIZE] = 1

U = np.zeros((MESH_T_SIZE + 1, MESH_X_SIZE + 1))

# начальные условия
U[0] = np.array([ICS(k * h) for k in range(MESH_X_SIZE + 1)])

t1 = 0
for j in range(1, MESH_T_SIZE + 1):
    B = np.zeros((MESH_X_SIZE + 1, 1))
    B1 = np.zeros((MESH_X_SIZE + 1, 1))
    F = np.zeros((MESH_X_SIZE + 1, 1))
    x1 = 0
    for i in range(1, MESH_X_SIZE):
        F[i, 0] = tau * f(t1, x1)
        x1 += h
    B[0, 0] = beta0(t1)
    B[MESH_X_SIZE, 0] = beta1(t1)
    t1 += tau
    B1[0, 0] = beta0(t1)
    B1[MESH_X_SIZE, 0] = beta1(t1)
    U[j] = (np.dot(inv(H), ((U[j - 1]).reshape((MESH_X_SIZE + 1, 1)) - B + B1 + F))).reshape((MESH_X_SIZE + 1))

u = U[100]
plt.plot(np.linspace(0, 1, MESH_X_SIZE + 1), u)
plt.show()
