import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)
        self.x, self.y = self.create_mesh()

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        Nx = self.px.N
        Ny = self.py.N
        x = self.px.create_mesh(Nx)
        y = self.py.create_mesh(Ny)
        self.x, self.y = np.meshgrid(x, y, indexing='ij')
        return self.x, self.y

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2x = self.px.D2()
        D2y = self.py.D2()
        return (sparse.kron(D2x, sparse.eye(self.py.N+1)) + sparse.kron(sparse.eye(self.px.N+1), D2y))

    def assemble(self, bcx, bcy, f=None):
        """Return assemble coefficient matrix A and right hand side vector b"""
        D = self.laplace()

        # Helper matrix B
        B = np.ones((self.px.N+1, self.py.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0

        bnds = np.where(B.ravel() == 1)[0]
        D = D.tolil()
        for i in bnds:
            D[i] = 0
            D[i, i] = 1

        b = np.zeros((self.px.N+1, self.py.N+1))
        b[1:-1, 1:-1] = sp.lambdify((x, y), f)(self.x[1:-1, 1:-1], self.y[1:-1, 1:-1])
        b[0,:] = sp.lambdify(y, bcx[0])(self.py.x)
        b[-1,:] = sp.lambdify(y, bcx[1])(self.py.x)
        b[:,0] = sp.lambdify(x, bcy[0])(self.px.x)
        b[:,-1] = sp.lambdify(x, bcy[1])(self.px.x)
        b = b.ravel()

        return D.tocsr(), b

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        uj = sp.lambdify((x, y), ue)(self.x, self.y)
        return np.sqrt(self.px.dx*self.py.dx*np.sum((uj-u)**2))

    def __call__(self, bcx=None, bcy=None, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(bcx=bcx, bcy=bcy, f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def test_poisson2d():
    Lx = 2; Ly = 2; N = 100
    sol = Poisson2D(Lx, Ly, N, N)

    ue = (3*x**2 - 2*x)*(y**2 - 4*y) + 3
    bcx = (ue.subs(x, 0), ue.subs(x, Lx))
    bcy = (ue.subs(y, 0), ue.subs(y, Ly))
    u = sol(bcx, bcy, f=(sp.diff(ue, x, 2) + sp.diff(ue, y, 2)))
    assert sol.l2_error(u, ue) < 1e-12

if __name__ == "__main__":
    test_poisson2d()

