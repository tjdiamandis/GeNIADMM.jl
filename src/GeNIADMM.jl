module GeNIADMM

using Random
using LinearAlgebra, SparseArrays, SuiteSparse
using Printf
using Krylov: CgSolver, cg!, issolved, warm_start!
using RandomizedPreconditioners
using StaticArrays
using LogExpFunctions: log1pexp, logistic
using Optim, LineSearches

const RP = RandomizedPreconditioners

include("linsys.jl")
include("types.jl")
include("utils.jl")
include("admm.jl")

end
