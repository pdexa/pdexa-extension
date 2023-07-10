# PDExa Extension

This repository contains extensions to Deal.II/Ginkgo/... that are necessary
for the PDExa project.

## Deal.II Extensions

### Ginkgo Wrappers

Wrappers for the following Ginkgo types are available:

- `matrix::Dense` aka Vector, this is subject of change due to ongoing deal.ii interface discussions
- Sparse matrix classes: `Csr, Coo, Ell, Hybrid, Sellp`
- Iterative Solver: `(F)Cg, Bicgstab, Cgs, (F)Gmres, Ir`
- Preconditioner: `Jacobi`