import numpy as np


def euler_method(boundary_arr, func_arr, h, n, condition=lambda a: False):
    result = boundary_arr
    for i in range(1, n):
        next_i = np.add(result[i-1], (h * func_arr(i-1, result)))
        if condition(next_i[0][0]):
            break
        result = np.append(result, next_i, axis=0)
    return result


def finite_difference_second_order(a, b, c, f, x_0, y_0, deriv_0, h, n, condition=lambda a: False):
    y = np.array([y_0 - h*deriv_0, y_0])
    for i in range(2,n-2):
        x_i = x_0+h*(i-1)
        a = get_func_val(a, x_i)
        b = get_func_val(b, x_i)
        c = get_func_val(c, x_i)
        f = get_func_val(f, x_i)
        a_0 = 2*a+b*h
        a_1 = (4*a-2*h*h*c)
        a_2 = 2*a-b*h
        a_3 = 2*h*h
        next_y = (y[i-1]*a_1 - y[i-2]*a_2 + a_3*f)/a_0
        if condition(next_y):
            break
        y = np.append(y, next_y)
    return y


def get_func_val(f, x):
    if callable(f):
        return f(x)
    else:
        return f


def wave_equation(f, g, c, max_h, n_t, n_h):
    tau = max_h
    h = c/tau
    while h >= max_h:
        tau /= 10
        h /= 10
    y = np.zeros(shape=(n_h, n_t))
    for i in range(2, n_h):
        y[0, i] = get_func_val(f, h*(i-1)) - h*get_func_val(g, h*(i-1))
        y[1, i] = get_func_val(f, h*(i-1))

    for j in range(2, n_t):
        for i in range(2, n_h):
            y[i, j-1] = y[i-1, j-2] + y[i-1, j] - y[i-2, j-1]
    return y
