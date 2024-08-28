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

dt_list = [(0.8, "#4E79A7"), (0.5, "#F28E2B"), (0.2, "#76B7B2"), (0.1,"#E15759") , (0.01,"#B07AA1")]

fig, axs = plt.subplots(1, 3, figsize=(16, 9))

for dt in dt_list:
    FE, t = solver(I=1, a=-1, T=8, dt=dt[0], theta=0)    # Forward Euler schemes
    axs[0].plot(t, FE, label=f"dt = {dt[0]}", color=dt[1])
    axs[0].set_title("Forward Euler schemes (theta = 0)")
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('Solution')
    axs[0].set_ylim(0, 5000)
    axs[0].legend()
    axs[0].grid(True)

for dt in dt_list:
    CN, t = solver(I=1, a=-1, T=8, dt=dt[0], theta=0.5)  # Crank-Nicolson schemes
    axs[1].plot(t, CN, label=f"dt = {dt[0]}", color=dt[1])
    axs[1].set_title('Crank-Nicolson schemes (theta = 0.5)')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Solution')
    axs[1].set_ylim(0, 5000)
    axs[1].legend()
    axs[1].grid(True)

for dt in dt_list:
    BE, t = solver(I=1, a=-1, T=8, dt=dt[0], theta=1)    # Backward Euler schemes
    axs[2].plot(t, BE, label=f"dt = {dt[0]}", color=dt[1])
    axs[2].set_title('Backward Euler schemes (theta = 1)')
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('Solution')
    axs[2].set_ylim(0, 5000)
    axs[2].legend()
    axs[2].grid(True)

# Adjust layout and show the figure
plt.tight_layout()
plt.show()