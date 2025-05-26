We are solving a Laplace equation with 2 Dirichlet boundary conditions and 4 Neumann boundary conditions:

$$\nabla^2\text{C}=0$$
$$\text{Dirichlet BCs:}\space\space\space\text{C(}x\text{=0)}=0\space\text{,}\space\space\text{C(}x\text{=L}_x\text{)}=1$$
$$\text{Neumann BCs:}\space\space\space\frac{\partial\text{C}}{\partial\text{n}}=0\space\space\in\space\space y=0\text{,}\space z=0\text{,}\space y=L_y\text{,}\space z=L_z$$
______________________________________________________________________
# Code Implementation

## 1. Import Statements

##### 🗃️ Data Imports & User Prompts

```.py
from read_leS import read_leS
from read_dat import read_dat

file_choice = input(".leS or .dat (L or D): ")
if file_choice == 'L':
    data = read_leS()
elif file_choice == 'D':
    data = read_dat()
```

Helper functions `read_leS()` and `read_dat()` are within separate code files that help import voxelized microstructure data. The user is prompted whether to import the .dat file or the .leS file. The resulting `data` is a NumPy array where column 4 indicates material (e.g. 0 = pore and 1 = solid).
##### 🔧 PETSc and DOLFINx

```.py
import importlib.util
import os
  
if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx
    from petsc4py import PETSc  # Correct import
  
    if not dolfinx.has_petsc:
        print("This code requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType  # type: ignore
else:
    print("This code requires petsc4py.")
    exit(0)
```

Checks whether or not PETSc is available. Imports `dolfinx` and `petsc4py` and related functions.
##### 🔸 Library Imports for FEM
```.py
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
```

Also importing standard tools to support FEM, such as parallel computing support (`MPI`) and symbolic math (UFL). From UFL, `dx`, `grad`, etc., are used in defining the PDE.

## 2. Mesh Setup

##### 📦 Mesh Generation

```.py
cell_dim = int(len(data[:,3]) ** (1/3)) + 1
n_x, n_y, n_z = (cell_dim - 1), (cell_dim - 1), (cell_dim - 1)
```

Defines how many cells (elements) are in each spatial direction. In this simulation, the mesh is defined using $n-1$ cells in each spatial direction rather than $n$ cells, where $n$ is the dimension of the imported voxelized data in each spatial direction. This choice is made to ensure that the number of mesh vertices, not cells, matches the number of voxels in the input microstructure data. 

```.py
msh = mesh.create_unit_cube(
    comm=MPI.COMM_WORLD,
    nx=n_x, ny=n_y, nz=n_z,
    cell_type=mesh.CellType.tetrahedron
)
```

Creates a unit cube mesh divided into tetrahedral elements. The mesh aligns with the resolution of the voxel grid.
##### 🧠 Function Space and Connectivity

```.py
V = fem.functionspace(msh, ("Lagrange", 1))
msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
```

Creates a scalar function space using linear Lagrange elements and builds mesh connectivity (e.g. between facets and cells) for boundary detection. See [[Revision Note - Connectivity]] for more information on this.

## 3. Implementing Boundary Conditions

```.py
facets_x0 = mesh.locate_entities_boundary(
    msh, dim=msh.topology.dim - 1, marker=lambda x: np.isclose(x[0], 0.0))
facets_xL = mesh.locate_entities_boundary(
    msh, dim=msh.topology.dim - 1, marker=lambda x: np.isclose(x[0], 1.0))
```

Locates the boundary facets where $x=0$ and $x=1$ to apply Dirichlet BCs.

```.py
facets_neumann = mesh.locate_entities_boundary(
    msh, dim=msh.topology.dim - 1,
    marker=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[2], 0.0) |
                     np.isclose(x[1], 1.0) | np.isclose(x[2], 1.0))
```

Locates the other four faces ($y=0$, $y=1$, $z=1$, and $z=1$), which implicitly get Neumann BCs (zero flux) in the weak form.

```.py
dofs_x0 = fem.locate_dofs_topological(V, entity_dim=2, entities=facets_x0, remote=True)
dofs_xL = fem.locate_dofs_topological(V, entity_dim=2, entities=facets_xL, remote=True)
bc_x0 = fem.dirichletbc(value=ScalarType(0), dofs=dofs_x0, V=V)
bc_xL = fem.dirichletbc(value=ScalarType(1), dofs=dofs_xL, V=V)
```

Finds the degrees of freedom (DOFs) associated with the $x=0$ and $x=1$ facets and then applies the Dirichlet BCs $C=0$ at inlet and $C=1$ at outlet.

## 4. Diffusivity and Definition of Variational Problem

##### 🧪 Diffusivity Field Setup

```.py
D_data = np.where(data[:,3] == 1, 0.0, 1.0)
V0 = fem.functionspace(msh, ("Lagrange", 1))
D = fem.Function(V0)
D.x.array[:] = D_data
```

This takes the microstructural data we imported earlier and converts material labels into binary diffusivity values where solid (1) becomes 0 diffusivity and pore (0) becomes 1 diffusivity. We then take this diffusivity array and create a `Function` to represent varying diffusivity `D(x)`.
##### 🧩 Define Variational (Weak) Form

```.py
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = D * inner(grad(v), grad(u)) * dx
L = fem.Constant(msh, PETSc.ScalarType(0)) * v * dx
```

This takes the standard FEM formulation of $\nabla \cdot (D(x) \nabla u)=0$ and turns it into its weak formulation, which is represented by the bilinear form in `a` and the linear form in `L`, where L is simply representative of the zero source term. See this article on the [weak form of Laplace's equation](https://scicomp.stackexchange.com/questions/37478/what-is-the-weak-form-of-a-vector-type-laplace-equation) for more information on how this equation was derived. The following shows the bilinear form:
$$a(u,v)=\int_{\Omega} D(x)\space (\nabla u\space\cdot\space\nabla v)\space dx$$
##### ⚙️ Solve Linear System

```.py
problem = LinearProblem(
    a, L, bcs=[bc_x0, bc_xL],
    petsc_options={"ksp_type": "bcgs", "pc_type": "gamg"})
uh = problem.solve()
```

The chosen solvers we use are the BiConjugate Gradient Stabilized (BiCGStab) method, which is good for us since it can handle non-symmetric and indefinite systems. We use Geometric-Algebraic Multigrid (GAMG) as a preconditioner to accelerate convergence. For a comprehensive list of solvers and preconditioners, please see: [PETSc User Manual](https://petsc.org/release/manual/). 

`problem.solve()` then solves the linear system $A\cdot u = b$ where $A$ is the matrix derived from `a`, $b$ is the vector derived from `L` and the resulting `uh` is the finite element function that satisfies the boundary conditions and the PDE.
##### 🧮 Compute Properties

```.py
porosity = np.count_nonzero(data[:,3]==0) / data[:,3].size
```

Computes porosity, which is a value that simply represents the percentage of electrolyte we have in our voxelized data.

```.py
Jx = 0
i = 0

for j in range(solution_size_cbrt):
    for k in range(solution_size_cbrt):
        index = i + j * solution_size_cbrt + k * solution_size_cbrt**2
        Jx += D_data[index] * (uh_x[index] - 1.0)

Jx /= solution_size
```

This approximates the average diffusive flux in the $x$-direction across the inlet face of the domain. The result `Jx` is normalized over the total number of degrees of freedom.

```.py
tortuosity = - porosity / (Jx * solution_size_cbrt)
```

Compute final value of tortuosity.
