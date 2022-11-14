from typing import *

import numpy as np


DEFAULT_KERNEL_FUNC: Callable[[float, float], float] = lambda x, y: x * y
DEFAULT_FUNC: Callable[[float], float] = lambda x: x


def _init_x_h(N: int, a: float, b: float) -> tuple[np.ndarray, float]:
    h = abs(b - a) / N
    x = np.array(range(N)) * h + h / 2 + a
    return x, h


def _init_a(
    N: int,
    kernel_func: Callable[[float, float], float],
    x: np.ndarray,
    h: float,
) -> np.matrix:
    kernel_func_h = lambda x, y: kernel_func(x, y) * h
    kernel = [[kernel_func_h(i, j) for j in x] for i in x]
    return np.identity(N) + np.matrix(kernel)


def init_equation(
    N: int,
    a: float = 0,
    b: float = 1,
    kernel_func: Callable[[float, float], float] = DEFAULT_KERNEL_FUNC,
    func: Callable[[float], float] = DEFAULT_FUNC,
) -> Callable[[], np.array]:
    x, h = _init_x_h(N, a, b)
    A = _init_a(N, kernel_func, x, h)
    B = func(x)
    return A, B
