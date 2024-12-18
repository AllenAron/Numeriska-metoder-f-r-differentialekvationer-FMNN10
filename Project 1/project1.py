import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def f(t: float, y) -> float:
    return 2*y

def sol(t:float, y0: float) -> float:
    return y0 * np.exp(2 * t)

def RK4step(f, t_old: float, y_old: float, h: float) -> float:

    Y1 = f(t_old, y_old)
    Y2 = f(t_old + h/2, y_old + h * Y1 / 2)
    Y3 = f(t_old + h/2, y_old + h * Y2 / 2)
    Y4 = f(t_old + h, y_old + h * Y3)

    y_new = y_old + h/6 * (Y1 + 2*Y2 + 2*Y3 + Y4)

    return y_new

def RK34step(f, t_old: float, y_old: list, h: float):

    Y1 = f(t_old, y_old)
    Y2 = f(t_old + h/2, y_old + np.multiply(h/2, Y1))
    Y3 = f(t_old + h/2, y_old +  np.multiply(h/2, Y2))
    Z3 = f(t_old + h, y_old -  np.multiply(h, Y1) +  np.multiply(2*h, Y2))
    Y4 = f(t_old + h, y_old +  np.multiply(h, Y3))

    y_new = y_old + h/6 * (Y1 +  np.multiply(2, Y2) +  np.multiply(2, Y3) + Y4)
    #z_new = y_old + h/6 * (Y1 + 4*Y2 + Z3)
    #l_new = z_new - y_new
    l_new = h/6 * ( np.multiply(2, Y2) + Z3 - np.multiply(2, Y3) - Y4)
    return y_new, l_new

def newstep(tol, err: float, err_old: float, h_old: float, k: int) -> float:
    h_new = np.abs((tol/err))**(2/(3*k)) * np.abs((tol/err_old))**(-1/(3*k)) * h_old
    return h_new


def RK4int(f, y0: float, t0: float, tf: float, N: int):
    h = (tf - t0) / N
    tgrid = np.linspace(t0, tf, N)

    
    #approx = RK4step(f, t0, y0, h)
    solution = y0
    approx = y0

    approx_list = []
    solution_list = []
    error_list = []

    approx_list.append(y0)
    #tgrid = np.insert(tgrid, 0, t0)
    #tgrid = np.delete(tgrid, -1)
    #print(tgrid)


    for t in tgrid:
        approx = RK4step(f, t, approx, h)
        approx_list.append(approx)
        solution = sol(t, y0)
        solution_list.append(solution)
        local_error = np.linalg.norm(approx - solution)
        error_list.append(local_error)

    
    #print(len(tgrid))
    #print(len(approx_list))
    approx_list = approx_list[:-1]
    return tgrid, approx_list, solution_list, error_list
    

def adaptiveRK34(fun, t0: float, tf: float, v0, tol: float):

    h = (np.abs(tf - t0) * tol**(1/4))/(100 * (1 + np.linalg.norm(fun(t0, v0))))
    t_list = []
    y_list = []
    t_list.append(h)
    y, l_new = RK34step(fun, t0, v0, h)
    y_list.append(y)
    l_old = tol
    nextstep = 0

    while(t_list[-1] + nextstep < tf):
        h = newstep(tol, np.linalg.norm(l_new), np.linalg.norm(l_old), h, 4)
        l_old = l_new
        y, l_new = RK34step(fun, t_list[-1], y, h)
        t_list.append(h + t_list[-1])
        y_list.append(y)
        nextstep = newstep(tol, np.linalg.norm(l_new), np.linalg.norm(l_old), h, 4)

    h = tf - t_list[-1]
    y, l_new = RK34step(fun, t_list[-1], y, h)
    t_list.append(h + t_list[-1])
    y_list.append(y)

    return t_list, y_list


def test():
    t0 = 0
    tf = 3
    y0 = 3
    
    N = 3
    h = (tf - t0) / N


    tgrid, approx_list, solution_list, error_list = RK4int(f, y0, t0, tf, N)

    exact_solution_list = []
    tgrid_solution = np.linspace(t0, tf, 300)
    
    for t in tgrid_solution:
        exact_solution_list.append(sol(t, y0))

    plt.plot(tgrid, approx_list)
    plt.plot(tgrid_solution, exact_solution_list)
    plt.legend(['Approximation', 'Solution'])
    plt.show()

    x = np.linspace(0, 10*np.pi, 100)
    y = np.sin(x)


    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'b-')


    xlist = []
    for p in x:
        xlist.append(p)
        line1.set_ydata(np.sin(xlist))

    for i in range(2, 30):
        x = np.linspace(t0, tf, i**2);
        tgrid, approx_list, solution_list, error_list = RK4int(f, y0, t0, tf, i**2)
        #ax.set_xlim(0, i)
        ax.cla()
        ax.plot(x, approx_list)
        ax.plot(tgrid_solution, exact_solution_list)
        plt.pause(0.1)
        fig.canvas.flush_events()

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    hgrid = []
    global_errors = []

    for n in range(2, 20):
        N = 2**n
        h = (tf - t0) / N
        tgrid, approx_list, solution_list, error_list = RK4int(f, y0, t0, tf, N)
        hgrid.append(h)
        global_errors.append(error_list[-1])
    
    ax.loglog(hgrid, global_errors)

    ylist = []
    for i in hgrid:
        ylist.append(i**4)
    ax.loglog(hgrid, ylist)
    ax.legend(['Global error', 'O(h^4)'])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    t, y = adaptiveRK34(f, t0, tf, y0, 1.e-6)
    ax.plot(t, y)
    timesteps = [-0.2]*len(t)
    solution_list = []

    for i in np.linspace(t0, tf):
        solution_list.append(sol(i, y0))

    ax.plot(np.linspace(t0, tf), solution_list)
    ax.scatter(t, timesteps)
    ax.legend(['Approximation', 'Solution', 'Time-steps'])
    plt.show()

def H(u):
    a = 3
    b = 9
    c = 15
    d = 15
    return c*u[0] + b*u[1] - d*np.log(u[0]) - c*np.log(u[1])

def lotka(t, u: list):
    a = 3
    b = 9
    c = 15
    d = 15
    return [a*u[0] - b*u[0]*u[1], c*u[0]*u[1] - d*u[1]]

def LotkaVolterra(t0, tf, x0, y0, tol):
    t_grid, v_list = adaptiveRK34(lotka, t0, tf, [x0, y0], tol)
    y_list = []
    x_list = []

    for u in v_list:
        x_list.append(u[0])
        y_list.append(u[1])

    timesteps = [-0.2]*len(t_grid)
    #plt.scatter(t_grid, timesteps)
    plt.plot(x_list, y_list)
    plt.show()

    h_list = []
    for u in v_list:
        h_list.append(np.linalg.norm(np.divide(H(u), H([x0, y0])) - 1))

    plt.plot(t_grid, x_list)
    plt.plot(t_grid, y_list)
    plt.show()
    #plt.plot(t_grid, h_list)
    #plt.show()


def defpol(µ):
    return lambda t, u: [u[1], µ*(1 - u[0]**2)*u[1] - u[0]]


def pol(t, u):
    µ = 100
    return [u[1], µ*(1 - u[0]**2)*u[1] - u[0]]


def VanDerPol(t, v0, tol):
    t_grid, v_list = adaptiveRK34(pol, t[0], t[1], v0, tol)

    y1_list = []
    y2_list = []

    for u in v_list:
        y1_list.append(u[0])
        y2_list.append(u[1])
    
    plt.plot(y1_list, y2_list)
    plt.show()
    plt.plot(t_grid, y2_list)
    plt.show()


    µ = [10, 15, 22, 33, 47, 68, 100, 150, 220]

    nbr_steps = []

    for m in µ:
        t_grid, v_list = adaptiveRK34(defpol(m), t[0], 2*m, v0, tol)
        nbr_steps.append(len(t_grid))

        y1_list = []
        y2_list = []

        for u in v_list:
            y1_list.append(u[0])
            y2_list.append(u[1])
        
        plt.plot(y1_list, y2_list, label=m)
        #plt.show()
        #plt.plot(t_grid, y2_list)
        #plt.show()
    plt.legend()
    plt.show()
    plt.plot(µ, nbr_steps)
    plt.show()
    plt.loglog(µ, nbr_steps)
    plt.show()
        

#VanDerPol([0, 200], [1, 30], 1.e-6)


#LotkaVolterra(0, 20, 15, 4, 1.e-6)

test()