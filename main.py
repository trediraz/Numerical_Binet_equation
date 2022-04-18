import matplotlib.pyplot as plt
import numpy as np

import numerical_methods as nm


def f(j, arr):
    return np.array([[f_u(j, arr), f_psi(j, arr)]])


def f_psi(j, arr):
    return 1 / semi_latus_rectum - arr[j][0]


def f_u(j, arr):
    return arr[j][1]


def set_up_analytical_graph():
    e = (semi_latus_rectum - r_0) / r_0
    if e == 1:
        limit = 2 * np.pi
    elif e > 1:
        limit = np.pi - np.arccos(r_0 / (semi_latus_rectum - r_0)) if semi_latus_rectum != r_0 else 4 * np.pi
    else:
        limit = 4 * np.pi
    theta = np.arange(0, limit, 0.00001)
    r = semi_latus_rectum / (1 + ((semi_latus_rectum - r_0) / r_0) * np.cos(theta))
    return theta, r


fig, ax = plt.subplots(2, 2, subplot_kw={'projection': 'polar'})
fig.set_figwidth(10)
fig.set_figheight(10)

max_r = 10
semi_latus_rectum = 3
r_0 = 2

boundary_con = np.zeros(shape=(1, 2))
boundary_con[0][0] = 1/r_0
boundary_con[0][1] = 0

h = [0.1, 0.01, 0.001]
n = 10000

theta, r = set_up_analytical_graph()

ax[0, 0].set_title("Rozwiązanie analityczne")
ax[0, 0].plot(theta, r, 'k')

for i in range(len(h)):
    e = nm.euler_method(boundary_con, f, h[i], n, lambda x: x <= 0)
    r = 1 / e[:, 0]
    ni = len(r)
    theta_num = np.arange(0, ni * h[i], h[i])
    ax[int((i+1)/2), (i+1) % 2].plot(theta_num, r)
    ax[int((i+1)/2), (i+1) % 2].set_rmax(max_r)
    ax[int((i+1)/2), (i+1) % 2].set_title(f'Krok: {h[i]}')


ax[0, 0].set_rmax(max_r)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()
