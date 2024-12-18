import matplotlib.pyplot as plt
import numpy as np

## Part 1

def sol(x):
    #return np.exp(2*x)
    return x**2*np.cos(np.sin(x))

def f(x):
    #return 2**2 * np.exp(2*x)
    return (2-x**2*np.cos(x)**2)*np.cos(np.sin(x))+ x*np.sin(np.sin(x))*(x*np.sin(x)-4*np.cos(x))

def I(x):
    L = 10
    return 1.e-3 * (3 - 2*np.cos((np.pi * x)/ L ))

def uf(x):
    E = 1.9 *1.e11
    return M(x) / E*I(x)

def Mf(x):
    q = -50

def funToVec(f, grid):
    vfun = np.vectorize(f)
    return vfun(grid)

def twopBVP(fvec: list, alpha: float, beta: float, L: float, N: int):
    dx = L/(N + 1)

    A = (1 / dx**2) * ((-1 * np.diag(2 * np.ones(N))) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1))
    
    fvec[0] += -alpha / (dx**2)
    fvec[-1] += -beta / (dx**2)
    
    y = np.linalg.solve(A, fvec)
    y = np.insert(y, 0, alpha)
    y = np.append(y, beta)

    return y

def errVSh(sol, f, alpha, beta, L, N):
    dx = L/(N + 1)
    spatial_grid = np.linspace(0, L, N + 2)
    fvec = funToVec(f, spatial_grid[1:-1])
    y = twopBVP(fvec, alpha, beta, L, N)
    exact_solution = funToVec(sol, spatial_grid)
    local_error = np.linalg.norm(exact_solution - y)
    plt.plot(spatial_grid, exact_solution)
    plt.plot(spatial_grid, y)
    plt.legend(['Exact solution', 'Approximation'])
    plt.show()

    h_list = []
    error_list = []
    for n in range(4, 15):
        N = n**2
        dx = L / (N + 1)
        h_list.append(dx)
        spatial_grid = np.linspace(0, L, N + 2)
        fvec = funToVec(f, spatial_grid[1:-1])
        y = twopBVP(fvec, alpha, beta, L, N)
        exact_solution = funToVec(sol, spatial_grid)
        local_error = np.linalg.norm(exact_solution - y)
        error_list.append(np.linalg.norm(local_error) * np.sqrt(dx))
    y = np.multiply(h_list, h_list)
    plt.loglog(h_list, error_list, label = 'Error')
    plt.loglog(h_list, y, label = 'O(dx**2)')
    plt.xlabel('Size of dx')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


def beam(alpha: float, beta: float, N: int, L: int):
    spatial_grid = np.linspace(0, L, N + 2)
    dx = L/(N + 1)

    I = lambda x: 1.e-3 * (3 - 2*np.cos((np.pi * x)/ L) **12)
    Mbis = [-50*1000] * len(spatial_grid)
    Mbis = Mbis[1:-1]
    
    M = twopBVP(Mbis, alpha, beta, L, N)

    E = 1.9 *1.e11

    I = np.vectorize(I)
    I = I(spatial_grid)

    ubis = np.divide(M, np.multiply(E, I))
    ubis = ubis[1:-1]

    u = twopBVP(ubis, alpha, beta, L, N)
    plt.plot(spatial_grid, u)
    plt.show()
    print(min(u))


## Part 2


def sturmLiouville(N: int, L:int):
    dx = L / (N + 1)

    lower_diag = np.diag(np.ones(N - 1), -1)
    lower_diag[-1] = 2 * lower_diag[-1]
    A = (1 / dx**2) * ((-1 * np.diag(2 * np.ones(N))) + np.diag(np.ones(N - 1), 1) + lower_diag)
    eig = np.linalg.eig(A)
    
    return eig

def sturmLiouville_eigen(N: int, L: int):
    eig = sturmLiouville(N, L)
    spatial_grid = np.linspace(0, L, N)

    index = np.argsort(eig[0])[-3:]
    
    eig_values = []
    eig_fun = []

    for i in index:
        eig_values.append(eig[0][i])
        eig_fun.append(eig[1][:,i])
        
    for i in range(3):
        plt.plot(spatial_grid, eig_fun[i])
    plt.show()

    actual_eig_values = [-np.pi**2 / 4, -9 * np.pi**2 / 4, -25 * np.pi**2 / 4]

    eig_values_list = []
    error_list = []
    N_list = []
    eig_values = []

    for n in range(4, 15):
        N = n**2
        eig_values = sturmLiouville(N, 1)[0]
        eig_values = np.sort(eig_values)[::-1][:3]

        N_list.append(N)
        eig_values_list.append(eig_values)
    
        sub_list = []
        for i in range(len(eig_values)):
            sub_list.append(np.abs(eig_values[i] - actual_eig_values[i]))

        error_list.append(sub_list)
    
    N_list = np.array(N_list, dtype = 'float')

    plt.loglog(N_list, N_list ** -2)
    plt.loglog(N_list, error_list)
    plt.show()


def schrodinger(N:int, L:int, n:int):
    spatial_grid = np.linspace(0, L, N + 2)
    dx = L / (N + 1)

    V = lambda x: np.sin(np.pi*x**2) ** 2
    V = lambda x: np.exp(x)
    V = lambda x: 0
    V = np.vectorize(V)
    V = V(spatial_grid[1:-1])
    
    A = (1 / dx**2) * ((-1 * np.diag(2 * np.ones(N) + V)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1))
    eig = np.linalg.eig(A)
    
    index = np.argsort(eig[0])[-n:]
    
    eig_values = []
    eig_fun = []
    density_fun = []

    for i in index:
        eig_values.append(eig[0][i])
        eig_fun.append(eig[1][:,i])
        
    
    for i in range(len(index)):
        eig_fun[i] = np.append(eig_fun[i], 0)
        eig_fun[i] = np.insert(eig_fun[i], 0, 0)
        eig_fun[i] = np.divide(eig_fun[i], np.dot(eig_fun[i],eig_fun[i])) / np.sqrt(dx)
        density_fun.append(np.multiply(eig_fun[i], eig_fun[i]))
        eig_fun[i] = eig_fun[i] * -np.min(eig_values) / 6
        eig_fun[i] = eig_fun[i] - eig_values[i]
        plt.plot(spatial_grid, eig_fun[i], label = i)
        plt.plot(spatial_grid, [- eig_values[i]] * len(spatial_grid), color = (0.8,0.8,0.8))

    plt.plot([0] * 50, np.linspace(0, np.max(eig_fun)), color = (0,0,0), linewidth = 2.5)
    plt.plot([1] * 50, np.linspace(0, np.max(eig_fun)), color = (0,0,0), linewidth = 2.5)
    plt.plot(np.linspace(0, 1), [0] * 50, color = (0,0,0), linewidth = 2.5)
        
    plt.show()

    for i in range(len(index)):
        density_fun[i] = density_fun[i] * -np.min(eig_values) / 6
        density_fun[i] = density_fun[i] - eig_values[i]
        plt.plot(spatial_grid, density_fun[i])
        plt.plot(spatial_grid, [min(density_fun[i])] * len(spatial_grid), color = (0.8,0.8,0.8))

    plt.plot([0] * 50, np.linspace(0, np.max(density_fun)), color = (0,0,0), linewidth = 2.5)
    plt.plot([1] * 50, np.linspace(0, np.max(density_fun)), color = (0,0,0), linewidth = 2.5)
    plt.plot(np.linspace(0, 1), [0] * 50, color = (0,0,0), linewidth = 2.5)

    plt.show()

errVSh(sol, f, 0, np.pi **2 * 4, 2*np.pi, 100)
beam(0, 0, 999, 10)
sturmLiouville_eigen(499, 1)
schrodinger(500, 1, 5)