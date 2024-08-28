import numpy as np
import matplotlib.pyplot as plt

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

dt_list = [1, 0.8, 0.2, 0.1, 0.01]

for i in range(len(dt_list)):
    exec(f"FE{i}"), t = solver(I=1, a=-1, T=8, dt=dt_list[i], theta=0)    # Forward Euler schemes
    exec(f"CN{i}"), t = solver(I=1, a=-1, T=8, dt=dt_list[i], theta=0.5)  # Crank-Nicolson schemes
    exec(f"BE{i}"), t = solver(I=1, a=-1, T=8, dt=dt_list[i], theta=1)    # Backward Euler schemes

fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot Finite Difference Method solution
for i in range(len(dt_list)):
    axs[0].plot(t, exec(f"FE{i}"), label="Forward Euler schemes", color='b')
    axs[1].plot(t, exec(f"CN{i}"), label="Crank-Nicolson schemes", color='r')
    axs[2].plot(t, exec(f"BE{i}"), label="Backward Euler schemes", color='g')

axs[0].set_title("Forward Euler schemes (theta = 0)")
axs[0].set_xlabel('t')
axs[0].set_ylabel('Solution')
axs[0].legend()
axs[0].grid(True)

axs[1].set_title('Crank-Nicolson schemes (theta = 0.5)d')
axs[1].set_xlabel('t')
axs[1].set_ylabel('Solution')
axs[1].legend()
axs[1].grid(True)

axs[2].set_title('Backward Euler schemes (theta = 1)')
axs[2].set_xlabel('t')
axs[2].set_ylabel('Solution')
axs[2].legend()
axs[2].grid(True)

# Adjust layout and show the figure
plt.tight_layout()
plt.show()



"""
from scitools.std import *

def A_exact(p):
    return np.exp(-p)

def A(p, theta):
    return (1-(1-theta)*p)/(1+theta*p)

def amplification_factor(names):
    curves = {}
    p = np.linspace(0, 3, 15)
    curves['exact'] = A_exact(p)
    plt.plot(p, curves['exact'])
    plt.hold('on')
    name2theta = dict(FE=0, BE=1, CN=0.5)
    for name in names:
        curves[name] = A(p, name2theta[name])
        plot(p, curves[name])
    plt.plot([p[0], p[-1]], [0, 0], '--')  # A=0 line
    plt.title('Amplification factors')
    plt.grid('on')
    plt.legend(['exact'] + names, loc='lower left', fancybox=True)
    plt.xlabel('$p=a\Delta t$')
    plt.ylabel('Amplification factor')
    plt.savefig('A_factors.png')
    plt.savefig('A_factors.pdf')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else: # default
        names = 'FE BE CN'.split()
    amplification_factor(names)"""