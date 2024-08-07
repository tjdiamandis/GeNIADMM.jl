# GeNIADMM

[![Build Status](https://github.com/tjdiamandis/GeNIADMM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tjdiamandis/GeNIADMM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/tjdiamandis/GeNIADMM.jl/branch/main/graph/badge.svg?token=QC5TKORCG1)](https://codecov.io/gh/tjdiamandis/GeNIADMM.jl)

This package contains the code to reproduce the figures in 
[On the (linear) convergence of Generalized Newton Inexact ADMM](https://arxiv.org/abs/2302.03863). 

You can add the package locally directly from this repo:
```julia
using Pkg
Pkg.add(url="https://github.com/tjdiamandis/GeNIADMM.jl")
```

All numerical experiments can be found in the `experiments` folder.

For a full solver that leverages the same ideas, check out [GeNIOS.jl](https://github.com/tjdiamandis/GeNIOS.jl).
