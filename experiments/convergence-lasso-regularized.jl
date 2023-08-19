using Pkg
Pkg.activate(joinpath(@__DIR__))
using OpenML, Tables, JLD2
using Random, LinearAlgebra, SparseArrays
using Plots, LaTeXStrings
using JuMP, MosekTools

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIADMM

const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE = joinpath(DATAPATH, "real-sim.jld2")

# Set this to false if you have not yet downloaded the real-sim dataset
const HAVE_DATA = true

if !HAVE_DATA
    real_sim = OpenML.load(1578)

    b_full = real_sim.class
    A_full = sparse(Tables.matrix(real_sim)[:,1:end-1])
    jldsave(DATAFILE, A_full, b_full)
else
    A_full, b_full = load(DATAFILE, "A_full", "b_full")
end


m = 10_000
Random.seed!(1)
p = randperm(length(b_full))[1:m]
b = b_full[p]
A = A_full[p, :]

γmax = norm(A'*b, Inf)
γ = 0.05*γmax
μ = 1.0


## Find true optimal value
function build_sparse_lasso_model(A, b, γ, μ)
    m, n = size(A)
    model = Model()
    @variable(model, x[1:n])
    @variable(model, z[1:m])
    @constraint(model, A*x - b .== z)
    
    # Add ℓ1 regularization
    @variable(model, t[1:n])
    @constraint(model, t .>= x)
    @constraint(model, t .>= -x)
    
    # Define objective
    @objective(model, Min, 0.5 * z'*z + 0.5 * μ * x'*x + γ * sum(t))
    return model
end

lasso_model = build_sparse_lasso_model(A/sqrt(m), b/sqrt(m), γ/m, μ/m)
set_optimizer(lasso_model, Mosek.Optimizer)
set_optimizer_attribute(lasso_model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-14)
set_optimizer_attribute(lasso_model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-14)
set_optimizer_attribute(lasso_model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-14)
set_optimizer_attribute(lasso_model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-14)
set_optimizer_attribute(lasso_model, "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL", 1.0)
set_optimizer_attribute(lasso_model, "MSK_IPAR_PRESOLVE_USE", 0)

JuMP.optimize!(lasso_model)
@show termination_status(lasso_model)
pstar = m*objective_value(lasso_model)

# zero out coefficients that are very close to 0 -> smaller obj value
zstar = value.(lasso_model[:x])
zstar[abs.(zstar) .< 1e-3] .= 0
pstar = 0.5 * norm(A*zstar - b,2)^2 + 0.5 * μ * norm(zstar,2)^2 + γ * norm(zstar,1)


## Solve
prob = GeNIADMM.LassoSolver(A, b, γ; ρ=10.0, μ=1.0)
res_gd = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=false, verbose=true, print_iter=100, gd_x_update=true,
    rho_update_iter=1000, multithreaded=true
)

prob = GeNIADMM.LassoSolver(A, b, γ; ρ=10.0, μ=1.0)
res_dir = GeNIADMM.solve!(
    prob; indirect=false, relax=false, max_iters=1000, tol=1e-4, logging=true,
    precondition=false, verbose=true, print_iter=100, rho_update_iter=1000,
    multithreaded=true
)

prob = GeNIADMM.LassoSolver(A, b, γ; ρ=10.0, μ=1.0)
res_nys = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=true, verbose=true, print_iter=100, rho_update_iter=1000,
    multithreaded=true, summable_step_size=true
)

prob = GeNIADMM.LassoSolver(A, b, γ; ρ=10.0, μ=1.0)
res_sketch = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=true, verbose=true, print_iter=100, sketch_solve_x_update=true,
    sketch_rank=500, rho_update_iter=1000, multithreaded=true
)

log_gd = res_gd.log
log_dir = res_dir.log
log_nys = res_nys.log
log_sketch = res_sketch.log

## Plots
# - dual_gap, rp, rd, obj_val
# - iter_time, linsys_time, precond_time, setup_time, solve_time
function add_to_plot!(plt, x, y, label, color; style=:solid, lw=3)
    start = findfirst(y[1:end-1] .> 0 .&& y[2:end] .> 0)
    inds = start:length(x)
    plot!(plt, x[inds], y[inds],
    label=label,
    lw=lw,
    linecolor=color,
    linestyle=style
)
end

FIGS_PATH = joinpath(@__DIR__, "figs")

rp_iter_plt = plot(; 
    dpi=300,
    yaxis=:log,
    ylabel=L"Primal Residual $\ell_2$ Norm",
    xlabel="Iteration",
    legend=:topright,
    legendfontsize=14,
    labelfontsize=14,
)
add_to_plot!(rp_iter_plt, 1:length(log_gd.iter_time), log_gd.rp, "Gradient", :coral)
add_to_plot!(rp_iter_plt, 1:length(log_sketch.iter_time), log_sketch.rp, "Sketch", :purple)
add_to_plot!(rp_iter_plt, 1:length(log_dir.iter_time), log_dir.rp, "ADMM (exact)", :red)
add_to_plot!(rp_iter_plt, 1:length(log_nys.iter_time), log_nys.rp, "NysADMM", :mediumblue; style=:dash)
savefig(rp_iter_plt, joinpath(FIGS_PATH, "lasso-rp-smooth.pdf"))

rd_iter_plt = plot(; 
    dpi=300,
    yaxis=:log,
    ylabel=L"Dual Residual $\ell_2$ Norm",
    xlabel="Iteration",
    legend=:topright,
    legendfontsize=14,
    labelfontsize=14,
)
add_to_plot!(rd_iter_plt, 1:length(log_gd.iter_time), log_gd.rd, "Gradient", :coral)
add_to_plot!(rd_iter_plt, 1:length(log_sketch.iter_time), log_sketch.rd, "Sketch", :purple)
add_to_plot!(rd_iter_plt, 1:length(log_dir.iter_time), log_dir.rd, "ADMM (exact)", :red)
add_to_plot!(rd_iter_plt, 1:length(log_nys.iter_time), log_nys.rd, "NysADMM", :mediumblue; style=:dash)
savefig(rd_iter_plt, joinpath(FIGS_PATH, "lasso-rd-smooth.pdf"))

obj_val_iter_plt = plot(; 
    dpi=300,
    yaxis=:log,
    ylabel=L"$\left| p - p^\star\right|/p^\star$",
    xlabel="Iteration",
    legend=:topright,
    legendfontsize=14,
    labelfontsize=14,
    # ylims=(1e-8, 100)
)
add_to_plot!(obj_val_iter_plt, 1:length(log_gd.iter_time), abs.(log_gd.obj_val .- pstar)./pstar, "Gradient", :coral)
add_to_plot!(obj_val_iter_plt, 1:length(log_sketch.iter_time), abs.(log_sketch.obj_val .- pstar)./pstar, "Sketch", :purple)
add_to_plot!(obj_val_iter_plt, 1:length(log_dir.iter_time), abs.(log_dir.obj_val .- pstar)./pstar, "ADMM (exact)", :red)
add_to_plot!(obj_val_iter_plt, 1:length(log_nys.iter_time), abs.(log_nys.obj_val .- pstar)./pstar, "NysADMM", :mediumblue; style=:dash)
# add_to_plot!(obj_val_iter_plt, 1:length(log_nys.iter_time), (log_nys.obj_val .- pstar_nys)./pstar_nys, "ADMM, Nystrom (pstar Nys)", :mediumblue)
savefig(obj_val_iter_plt, joinpath(FIGS_PATH, "lasso-obj-val-smooth.pdf"))
