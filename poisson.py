# Must use WSL for this code to function

# Import leS data
from read_leS import read_leS
from read_dat import read_dat

file_choice = input(".leS or .dat (L or D): ")
if file_choice == 'L':
    data = read_leS()
elif file_choice == 'D':
    data = read_dat()

# Library imports
import importlib.util

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx
    from petsc4py import PETSc  # Correct import

    if not dolfinx.has_petsc:
        print("This code requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType  # type: ignore
else:
    print("This code requires petsc4py.")
    exit(0)

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

# Create a 3D cubic mesh with tetrahedra cells
cell_dim = int(len(data[:,3]) ** (1/3)) + 1
n_x, n_y, n_z = (cell_dim - 1), (cell_dim - 1), (cell_dim - 1)  # Number of cells in each direction

msh = mesh.create_unit_cube(
    comm=MPI.COMM_WORLD,
    nx=n_x, 
    ny=n_y, 
    nz=n_z,  # Number of cells in x, y, and z directions
    cell_type=mesh.CellType.tetrahedron,  # Tetrahedral mesh
)

# Define function space (fixed function name)
V = fem.functionspace(msh, ("Lagrange", 1))

msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)

# Locate boundary facets for Dirichlet conditions at x=0 and x=Lx
facets_x0 = mesh.locate_entities_boundary(
    msh,
    dim=msh.topology.dim - 1,  # Boundary facets have dimension (topological dimension - 1)
    marker=lambda x: np.isclose(x[0], 0.0),  # x=0 plane
)
facets_xL = mesh.locate_entities_boundary(
    msh,
    dim=msh.topology.dim - 1,
    marker=lambda x: np.isclose(x[0], 1.0),  # x=Lx plane
)

# Neumann boundary conditions (dC/dn = 0) on other four planes
facets_neumann = mesh.locate_entities_boundary(
    msh,
    dim=msh.topology.dim - 1,
    marker=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[2], 0.0) |
                     np.isclose(x[1], 1.0) | np.isclose(x[2], 1.0),  # y=0, z=0, y=Ly, z=Lz
)

# Apply Dirichlet BCs at x=0 and x=Lx
dofs_x0 = fem.locate_dofs_topological(V, entity_dim=2, entities=facets_x0, remote=True)
dofs_xL = fem.locate_dofs_topological(V, entity_dim=2, entities=facets_xL, remote=True)
bc_x0 = fem.dirichletbc(value=ScalarType(0), dofs=dofs_x0, V=V)
bc_xL = fem.dirichletbc(value=ScalarType(1), dofs=dofs_xL, V=V)

# Set up diffusivity vector
D_data = np.where(data[:,3] == 1, 0.0, 1.0)
V0 = fem.functionspace(msh, ("Lagrange", 1))
D = fem.Function(V0)
D.x.array[:] = D_data

# Define weak formulation
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = D * ufl.inner(ufl.grad(v), ufl.grad(u)) * dx  # Bilinear form
L = fem.Constant(msh, PETSc.ScalarType(0)) * v * dx

# Solve the problem - note that Neumann conditions are normally included in the weak form via boundary integrals, so we do not need to include them
problem = LinearProblem(a, L, bcs=[bc_x0, bc_xL], petsc_options={"ksp_type": "bcgs", "pc_type": "gamg"})
uh = problem.solve()

uh_x = uh.x.array # Solution vector

# Compute porosity
porosity = np.count_nonzero(data[:,3]==0) / data[:,3].size

# flux_form = fem.form(- D * ufl.grad(uh)[0] * dx)
# Jx = fem.assemble_scalar(flux_form)

solution_size = uh_x.size
solution_size_cbrt = int(solution_size ** (1/3)) + 1
    
Jx = 0
i = 0
for j in range(solution_size_cbrt):
    for k in range(solution_size_cbrt):
            
        index = i + j * solution_size_cbrt + k * solution_size_cbrt**2

        Jx += D_data[index] * (uh_x[index] - 1.0)

Jx /= solution_size

print("Jx =", Jx)

print("Porosity =", porosity)

tortuosity = - porosity / (Jx * solution_size_cbrt)
print("Tortuosity =", tortuosity)
