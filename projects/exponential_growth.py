import numpy as np
import matplotlib.pyplot as plt
#from scitools.std import *
import sys

def solver(I, a, T, dt, theta):
    """Solve u'=-a*u, u(0)=I, for t in (0, T] with steps of dt."""
    Nt = int(T/dt)            # no of time intervals
    T = Nt*dt                 # adjust T to fit time step dt
    u = np.zeros(Nt+1)           # array of u[n] values
    t = np.linspace(0, T, Nt+1)  # time mesh
    u[0] = I                  # assign initial condition
    for n in range(0, Nt):    # n=0,1,...,Nt-1
        u[n+1] = (1 - (1-theta)*a*dt)/(1 + theta*dt*a)*u[n]
    return u, t

def exact(t, I=1, a=-1):
    return I*np.exp(-a*t)

def A_exact(p):
    return np.exp(-p)

def A(p, theta):
    return (1 - (1-theta)*p)/(1 + theta*p)

def amplification_factor(names):
    curves = {}
    p = np.linspace(-3, 0, 16)
    curves['exact'] = A_exact(p)
    plt.plot(p, curves['exact'], color="black", linestyle="--")
    #hold('on')
    name2theta = dict(FE=0, BE=1, CN=0.5)
    for name in names:
        curves[name] = A(p, name2theta[name])
        plt.plot(p, curves[name])
    #plt.plot([p[0], p[-1]], [0, 0], '--')  # A=0 line
    plt.title('Amplification factors')
    plt.grid('on')
    plt.legend(['Exact'] + names, loc='lower right', fancybox=True)
    plt.xlabel(r'$p=a\Delta t$')
    plt.ylabel('Amplification factor')
    plt.show()

if __name__ == '__main__':

    dt_list = [(1.25, "#ffc100"), (0.8, "#c356ea"), (0.6, "#8ff243"), (0.2,"#71aef2") , (0.1,"#ea5645")]

    fig, axs = plt.subplots(1, 3, figsize=(16, 9))

    for dt in dt_list:
        FE, t = solver(I=1, a=-1, T=8, dt=dt[0], theta=0)    # Forward Euler schemes
        axs[0].plot(t, FE, label=rf"$\Delta$t = {dt[0]}", color=dt[1])

    axs[0].plot(t, exact(t), label="Exact solution", color="black",linestyle="--")
    axs[0].set_title(r"Forward Euler schemes ($\theta = 0$)")
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('Solution')
    axs[0].set_ylim(-1000, 4000)
    axs[0].legend()
    axs[0].grid(True)


    for dt in dt_list:
        CN, t = solver(I=1, a=-1, T=8, dt=dt[0], theta=0.5)  # Crank-Nicolson schemes
        axs[1].plot(t, CN, label=rf"$\Delta$t = {dt[0]}", color=dt[1])

    axs[1].plot(t, exact(t), label="Exact solution", color="black",linestyle="--")
    axs[1].set_title(r'Crank-Nicolson schemes ($\theta = 0.5$)')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Solution')
    axs[1].set_ylim(-1000, 4000)
    axs[1].legend()
    axs[1].grid(True)

    for dt in dt_list:
        BE, t = solver(I=1, a=-1, T=8, dt=dt[0], theta=1)    # Backward Euler schemes
        axs[2].plot(t, BE, label=rf"$\Delta$t = {dt[0]}", color=dt[1])

    axs[2].plot(t, exact(t), label="Exact solution", color="black",linestyle="--")
    axs[2].set_title(r'Backward Euler schemes ($\theta = 1$)')
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('Solution')
    axs[2].set_ylim(-1000, 4000)
    axs[2].legend()
    axs[2].grid(True)

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else: # default
        names = 'FE BE CN'.split()
    amplification_factor(names)