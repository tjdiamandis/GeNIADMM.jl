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
const SAVEFILE = joinpath(DATAPATH, "lasso.jld2")

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


## Find true optimal value
function build_sparse_lasso_model(A, b, γ)
    m, n = size(A)
    model = Model()
    @variable(model, x[1:n])
    @variable(model, z[1:m])
    @constraint(model, A*x - b .== z)
    
    # Add ℓ1 regularization
    @variable(model, 0.0 <= reg)
    @constraint(model, [reg; x] in MOI.NormOneCone(n + 1))
    
    # Define objective
    @objective(model, Min, 0.5 * z'*z + γ * reg)
    return model
end

lasso_model = build_sparse_lasso_model(A/sqrt(m), b/sqrt(m), γ/m)
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
pstar = 0.5 * norm(A*zstar - b,2)^2 + γ * norm(zstar,1)


## Solve
prob = GeNIADMM.LassoSolver(A, b, γ; ρ=1.0)
res_gd = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=false, verbose=true, print_iter=100, gd_x_update=true,
    rho_update_iter=1000, multithreaded=true
)

prob = GeNIADMM.LassoSolver(A, b, γ; ρ=1.0)
res_agd = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=false, verbose=true, print_iter=100, agd_x_update=true,
    rho_update_iter=1000, multithreaded=true
)

prob = GeNIADMM.LassoSolver(A, b, γ; ρ=1.0)
res_opt = GeNIADMM.solve!(
    prob; indirect=true, relax=true, max_iters=1000, tol=1e-10, logging=true,
    precondition=true, verbose=true, print_iter=100, rho_update_iter=1000,
    multithreaded=true, summable_step_size=true
)

prob = GeNIADMM.LassoSolver(A, b, γ; ρ=1.0)
res_dir = GeNIADMM.solve!(
    prob; indirect=false, relax=false, max_iters=1000, tol=1e-4, logging=true,
    precondition=false, verbose=true, print_iter=100, rho_update_iter=1000,
    multithreaded=true
)

prob = GeNIADMM.LassoSolver(A, b, γ; ρ=1.0)
res_nys = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=true, verbose=true, print_iter=100, rho_update_iter=1000,
    multithreaded=true, summable_step_size=true
)

prob = GeNIADMM.LassoSolver(A, b, γ; ρ=1.0)
res_sketch = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=true, verbose=true, print_iter=100, sketch_solve_x_update=true,
    sketch_rank=500, rho_update_iter=1000, multithreaded=true
)

prob = GeNIADMM.LassoSolver(A, b, γ; ρ=1.0)
res_sketch_no_correction = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=true, verbose=true, print_iter=100, sketch_solve_x_update=true,
    sketch_rank=500, rho_update_iter=1000, multithreaded=true, add_Enorm=false
)

save(SAVEFILE, 
    "res_gd", res_gd,
    "res_agd", res_agd,
    "res_dir", res_dir,
    "res_nys", res_nys,
    "res_sketch", res_sketch,
    "res_sketch_no_correction", res_sketch_no_correction,
    "res_opt", res_opt,
    "pstar", pstar
)

## Load data
res_gd, res_agd, res_dir, res_nys, res_sketch, res_sketch_no_correction, res_opt, pstar = load(
    SAVEFILE, 
    "res_gd",
    "res_agd",
    "res_dir",
    "res_nys",
    "res_sketch",
    "res_sketch_no_correction",
    "res_opt",
    "pstar"
)


log_gd = res_gd.log
log_agd = res_agd.log
log_dir = res_dir.log
log_nys = res_nys.log
log_sketch = res_sketch.log
log_sketch_no_correction = res_sketch_no_correction.log
log_opt = res_opt.log
pstar_nys = res_opt.obj_val
@show pstar_nys - pstar

## Plots
# - dual_gap, rp, rd, obj_val
# - iter_time, linsys_time, precond_time, setup_time, solve_time

function add_to_plot!(plt, x, y, label, color; style=:solid, lw=3, marker=:none, alpha=1.0)
    start = findfirst(y[1:end-1] .> 0 .&& y[2:end] .> 0)
    inds = start:length(x)
    plot!(plt, x[inds], y[inds],
    label=label,
    lw=lw,
    linecolor=color,
    linestyle=style,
    marker=marker,
    linealpha=alpha
)
end

FIGS_PATH = joinpath(@__DIR__, "figs")

dual_gap_iter_plt = plot(; 
    dpi=300,
    legendfontsize=12,
    labelfontsize=14,
    yaxis=:log,
    ylabel="Dual Gap",
    xlabel="Iteration",
    legend=:topright,
    # ylims=(9e-5, 1e3)
)
add_to_plot!(dual_gap_iter_plt, 1:length(log_gd.iter_time), log_gd.dual_gap, "Gradient", :coral)
add_to_plot!(dual_gap_iter_plt, 1:length(log_agd.iter_time), log_agd.dual_gap, "AGD", :turquoise)
add_to_plot!(dual_gap_iter_plt, 1:length(log_sketch.iter_time), log_sketch.dual_gap, "Sketch", :purple1)
add_to_plot!(dual_gap_iter_plt, 1:length(log_dir.iter_time), log_dir.dual_gap, "ADMM (exact)", :red)
add_to_plot!(dual_gap_iter_plt, 1:length(log_nys.iter_time), log_nys.dual_gap, "NysADMM", :mediumblue, style=:dash)
savefig(dual_gap_iter_plt, joinpath(FIGS_PATH, "lasso-dual-gap-updated.pdf"))

rp_iter_plt = plot(; 
    dpi=300,
    legendfontsize=12,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Primal Residual $\ell_2$ Norm",
    xlabel="Iteration",
    legend=:topright,
)
add_to_plot!(rp_iter_plt, 1:length(log_gd.iter_time), log_gd.rp, "Gradient", :coral)
add_to_plot!(rp_iter_plt, 1:length(log_agd.iter_time), log_agd.rp, "AGD", :turquoise)
add_to_plot!(rp_iter_plt, 1:length(log_sketch.iter_time), log_sketch.rp, "Sketch", :purple1)
add_to_plot!(rp_iter_plt, 1:length(log_dir.iter_time), log_dir.rp, "ADMM (exact)", :red)
add_to_plot!(rp_iter_plt, 1:length(log_nys.iter_time), log_nys.rp, "NysADMM", :mediumblue, style=:dash)
savefig(rp_iter_plt, joinpath(FIGS_PATH, "lasso-rp-updated.pdf"))

rd_iter_plt = plot(; 
    dpi=300,
    legendfontsize=12,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Dual Residual $\ell_2$ Norm",
    xlabel="Iteration",
    legend=:topright,
)
add_to_plot!(rd_iter_plt, 1:length(log_gd.iter_time), log_gd.rd, "Gradient", :coral)
add_to_plot!(rd_iter_plt, 1:length(log_agd.iter_time), log_agd.rd, "AGD", :turquoise)
add_to_plot!(rd_iter_plt, 1:length(log_sketch.iter_time), log_sketch.rd, "Sketch", :purple1)
add_to_plot!(rd_iter_plt, 1:length(log_dir.iter_time), log_dir.rd, "ADMM (exact)", :red)
add_to_plot!(rd_iter_plt, 1:length(log_nys.iter_time), log_nys.rd, "NysADMM", :mediumblue, style=:dash)
savefig(rd_iter_plt, joinpath(FIGS_PATH, "lasso-rd-updated.pdf"))

obj_val_iter_plt = plot(; 
    dpi=300,
    legendfontsize=12,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"$\left| p - p^\star\right|/p^\star$",
    xlabel="Iteration",
    legend=:topright,
    # ylims=(1e-8, 100)
)
add_to_plot!(obj_val_iter_plt, 1:length(log_gd.iter_time), abs.(log_gd.obj_val .- pstar)./pstar, "Gradient", :coral)
add_to_plot!(obj_val_iter_plt, 1:length(log_agd.iter_time), abs.(log_agd.obj_val .- pstar)./pstar, "AGD", :turquoise)
add_to_plot!(obj_val_iter_plt, 1:length(log_sketch.iter_time), abs.(log_sketch.obj_val .- pstar)./pstar, "Sketch", :purple1)
add_to_plot!(obj_val_iter_plt, 1:length(log_dir.iter_time), abs.(log_dir.obj_val .- pstar)./pstar, "ADMM (exact)", :red)
add_to_plot!(obj_val_iter_plt, 1:length(log_nys.iter_time), abs.(log_nys.obj_val .- pstar)./pstar, "NysADMM", :mediumblue, style=:dash)
savefig(obj_val_iter_plt, joinpath(FIGS_PATH, "lasso-obj-val-updated.pdf"))

lasso_plt = plot(; 
    dpi=300,
    # title="Convergence (Lasso)",
    yaxis=:log,
    xlabel="Iteration",
    legend=:topright,
    ylims=(1e-10, 1000),
    legendfontsize=12,
    labelfontsize=14,
    titlefontsize=14
)
add_to_plot!(lasso_plt, 1:length(log_opt.iter_time), log_opt.rp, "Primal Residual", :turquoise)
add_to_plot!(lasso_plt, 1:length(log_opt.iter_time), log_opt.rd, "Dual Residual", :red)
add_to_plot!(lasso_plt, 1:length(log_opt.iter_time), log_opt.dual_gap, "Duality Gap", :mediumblue)
add_to_plot!(lasso_plt, 1:length(log_opt.iter_time), sqrt(eps())*ones(length(log_opt.iter_time)), L"\sqrt{\texttt{eps}}", :black; style=:dash, lw=1)
savefig(lasso_plt, joinpath(FIGS_PATH, "lasso-updated.pdf"))

## Divergence plot
end_ind = findfirst(x-> x > 1e20, log_sketch_no_correction.rd)
divergence_plt = plot(; 
    dpi=300,
    yaxis=:log,
    xlabel="Iteration",
    legend=:topleft,
    legendfontsize=10,
    labelfontsize=14,
    # ylims=(1e-2, 5e2)
)
add_to_plot!(divergence_plt, 1:end_ind, log_sketch.rp[1:end_ind], "Primal Residual", :turquoise, lw=2)
add_to_plot!(divergence_plt, 1:end_ind, log_sketch.rd[1:end_ind], "Dual Residual", :red, lw=2)
add_to_plot!(divergence_plt, 1:end_ind, log_sketch.dual_gap[1:end_ind], "Duality Gap", :mediumblue, lw=2)
add_to_plot!(divergence_plt, 1:end_ind, log_sketch_no_correction.rp[1:end_ind], nothing, :turquoise, lw=2, style=:dash)
add_to_plot!(divergence_plt, 1:end_ind, log_sketch_no_correction.rd[1:end_ind], nothing, :red, lw=2, style=:dash)
add_to_plot!(divergence_plt, 1:end_ind, log_sketch_no_correction.dual_gap[1:end_ind], nothing, :mediumblue, lw=2, style=:dash)
savefig(divergence_plt, joinpath(FIGS_PATH, "lasso-divergence-updated.pdf"))

