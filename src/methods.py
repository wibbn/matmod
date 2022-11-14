from typing import *

import numpy as np


def simple_iteration(A: np.matrix, f: np.ndarray, epsilon: float = 0.0001) -> Generator[np.ndarray, None, None]:
    N = A.shape[0]
    u = np.random.random(N)
    u_old = np.random.random(N)

    B = np.eye(N) - A
    A = A / np.max(np.abs(A))
    f = f / np.max(np.abs(A))
    while np.linalg.norm(u - u_old) / np.linalg.norm(f) > epsilon:
        u_old = u
        u = np.dot(B, u) + f
        u = np.squeeze(np.asarray(u))
        yield u


def gradient_descent(A: np.matrix, B: np.ndarray, eps: float = 1e-6) -> Coroutine[np.ndarray, None, None]:
    N = A.shape[0]
    u = np.random.random(N)
    u_old = np.zeros(N)
    while np.linalg.norm(u - u_old) / np.linalg.norm(B) >= eps:
        u_old = u
        r = A @ u - B
        r = np.squeeze(np.asarray(r))
        A_star_r = np.squeeze(np.asarray(np.conj(A.T) @ r))
        u = u - np.linalg.norm(A_star_r) ** 2 / np.linalg.norm(A @ A_star_r) ** 2 * A_star_r
        u = np.squeeze(np.asarray(u))
        yield u

