# PDExa Extension

This repository contains extensions to Deal.II/Ginkgo/... that are necessary
for the PDExa project, as well as example applications.

## Deal.II Extensions

### Ginkgo Interface

The interface presents a lightweight approach to make use Ginkgo solvers and preconditioners in deal.II.
It provides:

- mappings from deal.II `LinearAlgebra::distributed::Vector` to Ginkgo `Dense` and `distributed::Vector` vectors,
- mappings from deal.II `SparseMatrix` to Ginkgo's sparse matrix types,
- mappings from Ginkgo's `LinOp` and `LinOpFactory` to deal.II `LinearOperator`.

The vector mappings are implemented without any copy operation.
The Ginkgo vectors will access the same memory as the deal.II vectors.
