import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def ieulerstep(A: np.ndarray, uold: np.ndarray, h: float) -> np.ndarray:
    # y_n+1 = y_n + h * f(t_n+1, y_n+1)
    # f(t_n+1, y_n+1) = A * y_n+1
    # (I - h*A) * y_n+1 = y_n
    # y_n+1 = (I - h*A)^-1 * y_n

    #if len(A) == 1:
     #   return 1/(1 - h*A[0][0]) * uold

    return np.linalg.inv(np.identity(len(A)) - np.multiply(h, A)) @ uold

def ieulerint(A: np.ndarray, y0: float, t0: float, tf: float, N: int):
    approx = ieulerstep(A, y0, t0)
    solution = y0
    tgrid = np.linspace(t0, tf, N)

    approx_list = []
    solution_list = []
    error_list = []
    relative_error = []
    
    for t in tgrid:
        approx = ieulerstep(A, approx, (tf - t0)/N)
        approx_list.append(approx)
        solution = linalg.expm(np.multiply(t,A)) @ y0
        solution_list.append(solution)
        error_list.append(np.linalg.norm(approx - solution))
        relative_error.append(np.linalg.norm(np.divide(approx - solution, np.linalg.norm(solution))))
    
    global_error = np.linalg.norm(approx - solution)

    return tgrid, approx_list, solution_list, error_list, relative_error

def ierrVSh(A, y0, t0, tf, N):
    x_list = []
    y_list = []
    rel_list = []
    for i in range(1, N):
        x_list.append((tf-t0)/(2*i))
        y_list.append(ieulerint(A, y0, t0, tf, 2*i)[3][-1])
        rel_list.append(ieulerint(A, y0, t0, tf, 2*i)[4][-1])
    plt.plot(x_list, y_list)
    plt.plot(x_list, rel_list)
    plt.legend(['y', 'rel'])


def test(A, y0, t0, tf, N):

    # global error as a function of stepsize for different lambda
    for i in range(-10, 10):
        ierrVSh(A, y0, 0, i + 10, i + 20)
        plt.legend(range(-10, 10))

    plt.show()

    plt.show()

    tgrid, approx_list, solution_list, error_list, relative_error = ieulerint(A, y0, t0, tf, N)
    plt.plot(tgrid, approx_list)
    plt.plot(tgrid, solution_list)
    plt.legend(['approximation of f1', 'approximation of f2', 'solution of f1', 'solution of f1'])
    plt.show()

test([[-1, 100], [0, -30]], [1, 1], 0, 10, 100)
