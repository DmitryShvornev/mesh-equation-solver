import json
from math import *
import matplotlib.pyplot as plt
import csv
from sympy import *
import numpy as np
from numpy.linalg import inv

# "ICS": "(x - 0.5)**2", //u(x,0)
# "BC": ["2", "3"], //u(0,t), u(1,t)

# Неявная схема

with open("task.json", 'r') as task_file:
    task = json.load(task_file)

print(task)

def f(x, t):
    return x * t

h = task["solver"]["h"]
tau = task["solver"]["tau"]
a = task["a"]
p = -a * tau / (h ** 2)
q = 1 + 2 * a * tau / (h ** 2)
r = p
MESH_SIZE = 100
beta0 = task["BC"][0]
beta1 = task["BC"][1]

H = np.zeros((MESH_SIZE + 1, MESH_SIZE + 1))
for i in range(1, H.shape[0] - 1):
    H[i, i-1] = p
    H[i, i] = q
    H[i, i+1] = r
H[0, 0] = 1
H[MESH_SIZE, MESH_SIZE] = 1
print(H)

U1 = np.zeros((1001, MESH_SIZE + 1))
U1[0] = np.array([2 * beta0 for k in range(MESH_SIZE + 1)])
t1 = 0
for j in range(1, 1001):
    B = np.zeros((MESH_SIZE + 1, 1))
    B1 = np.zeros((MESH_SIZE + 1, 1))
    F = np.zeros((MESH_SIZE + 1, 1))
    x1 = 0
    for i in range(1, MESH_SIZE):
        F[i, 0] = tau * f(t1, x1)
        x1 += h
    t1 += tau
    B[0, 0] = beta0
    B[MESH_SIZE, 0] = beta0
    B1[0, 0] = beta1
    B1[MESH_SIZE, 0] = beta1
    U1[j] = (np.dot(inv(H),((U1[j - 1]).reshape((MESH_SIZE + 1, 1)) - B1 + B + F))).reshape((MESH_SIZE + 1))


u3 = U1[500]
plt.plot(np.linspace(0,1,MESH_SIZE + 1),u3)
plt.show()
