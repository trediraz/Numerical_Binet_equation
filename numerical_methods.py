import numpy as np


def euler_method(boundary_arr, func_arr, h, n, condition):
    result = boundary_arr
    for i in range(1, n):
        next_i = np.add(result[i-1], (h * func_arr(i-1, result)))
        if condition(next_i[0][0]):
            break
        result = np.append(result, next_i, axis=0)
    return result
