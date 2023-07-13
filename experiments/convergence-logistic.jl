using Pkg
Pkg.activate(joinpath(@__DIR__))
using OpenML, Tables, JLD2, Printf
using Random, LinearAlgebra, SparseArrays
using Plots, LaTeXStrings
using JuMP, MosekTools

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIADMM

const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE = joinpath(DATAPATH, "real-sim.jld2")
const SAVEFILE = joinpath(DATAPATH, "logistic.jld2")

# Set this to false if you have not yet downloaded the real-sim dataset
const HAVE_DATA = true

if !HAVE_DATA
    real_sim = OpenML.load(1578)

    b_full = real_sim.class
    A_full = sparse(Tables.matrix(real_sim)[:,1:end-1])
    save(DATAFILE, "A_full", A_full, "b_full", b_full)
else
    A_full, b_full = load(DATAFILE, "A_full", "b_full")
end

m = 10_000
Random.seed!(1)
p = randperm(length(b_full))[1:m]
b = b_full[p]
A = A_full[p, :]
γmax = norm(0.5*A'*ones(m), Inf)
γ = 0.05*γmax


## Find ``true optimal''
# From https://jump.dev/JuMP.jl/stable/tutorials/conic/logistic_regression/
function softplus(model, t, u)
    z = @variable(model, [1:2], lower_bound = 0.0)
    @constraint(model, sum(z) <= 1.0)
    @constraint(model, [u - t, 1, z[1]] in MOI.ExponentialCone())
    @constraint(model, [-t, 1, z[2]] in MOI.ExponentialCone())
end

function build_sparse_logit_model(A, b, γ)
    n, p = size(A)
    model = Model()
    @variable(model, x[1:p])
    @variable(model, t[1:n])
    for i in 1:n
        u = -(A[i, :]' * x) * b[i]
        softplus(model, t[i], u)
    end
    # Add ℓ1 regularization
    @variable(model, 0.0 <= reg)
    @constraint(model, [reg; x] in MOI.NormOneCone(p + 1))
    # Define objective
    @objective(model, Min, sum(t) + γ * reg)
    return model
end

logistic_model = build_sparse_logit_model(A, b, γ)
set_optimizer(logistic_model, Mosek.Optimizer)
set_optimizer_attribute(logistic_model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-14)
set_optimizer_attribute(logistic_model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-14)
set_optimizer_attribute(logistic_model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-14)
set_optimizer_attribute(logistic_model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-14)
set_optimizer_attribute(logistic_model, "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL", 1.0)
set_optimizer_attribute(logistic_model, "MSK_IPAR_PRESOLVE_USE", 0)
JuMP.optimize!(logistic_model)
pstar = objective_value(logistic_model)


## Solve
prob = GeNIADMM.LogisticSolver(A, b, γ; ρ=1.0)
res_gd = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=false, verbose=true, print_iter=100, gd_x_update=true,
    rho_update_iter=1000, multithreaded=true
)

prob = GeNIADMM.LogisticSolver(A, b, γ; ρ=1.0)
res_exact = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=1000, tol=1e-4, logging=true,
    precondition=false, verbose=true, print_iter=1, rho_update_iter=1000,
    logistic_exact_solve=true, multithreaded=true
)

prob = GeNIADMM.LogisticSolver(A, b, γ; ρ=1.0)
res_nys = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=true, verbose=true, print_iter=100, rho_update_iter=1000,
    multithreaded=true
)

prob = GeNIADMM.LogisticSolver(A, b, γ; ρ=1.0)
res_nys_summable = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=true, verbose=true, print_iter=100, rho_update_iter=1000,
    multithreaded=true, summable_step_size=true
)

prob = GeNIADMM.LogisticSolver(A, b, γ; ρ=1.0)
res_sketch = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=false, verbose=true, print_iter=100, sketch_solve_x_update=true,
    sketch_rank=500, rho_update_iter=1000,
    multithreaded=true
)

prob = GeNIADMM.LogisticSolver(A, b, γ; ρ=1.0)
res_opt = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=1000, tol=1e-10, logging=true,
    precondition=true, verbose=true, print_iter=100, rho_update_iter=1000,
    multithreaded=true
)
pstar_nys = res_opt.obj_val

prob = GeNIADMM.LogisticSolver(A, b, γ; ρ=1.0)
res_sketch_no_correction = GeNIADMM.solve!(
    prob; indirect=true, relax=false, max_iters=500, tol=1e-4, logging=true,
    precondition=false, verbose=true, print_iter=1, sketch_solve_x_update=true,
    sketch_rank=500, rho_update_iter=1000,
    multithreaded=true, add_Enorm=false
)

save(SAVEFILE, 
    "res_gd", res_gd,
    "res_exact", res_exact,
    "res_nys", res_nys,
    "res_nys_summable", res_nys_summable,
    "res_sketch", res_sketch,
    "res_sketch_no_correction", res_sketch_no_correction,
    "res_opt", res_opt,
    "pstar", pstar
)

## Load data
res_gd, res_exact, res_nys, res_nys_summable, res_sketch, res_sketch_no_correction, res_opt, pstar = load(
    SAVEFILE, 
    "res_gd",
    "res_exact",
    "res_nys",
    "res_nys_summable",
    "res_sketch",
    "res_sketch_no_correction",
    "res_opt",
    "pstar"
)

log_gd = res_gd.log
log_nys = res_nys.log
log_nys_summable = res_nys_summable.log
log_exact = res_exact.log
log_sketch = res_sketch.log
log_sketch_no_correction = res_sketch_no_correction.log
log_opt = res_opt.log
@show pstar - res_opt.obj_val

# Printout timings
@printf("\nADMM, Exact:")
@printf("- setup:    %6.3f", log_exact.setup_time)
@printf("- iter:    %6.3f", log_exact.solve_time / length(log_exact.dual_gap))

@printf("\nNysADMM:")
@printf("- setup:    %6.3f", log_nys.setup_time)
@printf("- iter:    %6.3f", log_nys.solve_time / length(log_nys.dual_gap))

@printf("\nNysADMM (Summable):")
@printf("- setup:    %6.3f", log_nys_summable.setup_time)
@printf("- iter:    %6.3f", log_nys_summable.solve_time / length(log_nys_summable.dual_gap))

@printf("\nGradient Descent:")
@printf("- setup:    %6.3f", log_gd.setup_time)
@printf("- iter:    %6.3f", log_gd.solve_time / length(log_gd.dual_gap))

@printf("\nSketch & Solve:")
@printf("- setup:    %6.3f", log_sketch.setup_time)
@printf("- iter:    %6.3f", log_nys.solve_time / length(log_nys.dual_gap))



function add_to_plot!(plt, x, y, label, color; style=:solid, lw=3)
    start = findfirst(y .> 0)
    inds = start:length(x)
    plot!(plt, x[inds], y[inds],
    label=label,
    lw=lw,
    linecolor=color,
    linestyle=style
)
end

FIGS_PATH = joinpath(@__DIR__, "figs")

dual_gap_iter_plt = plot(; 
    dpi=300,
    legendfontsize=14,
    labelfontsize=14,
    yaxis=:log,
    ylabel="Dual Gap",
    xlabel="Iteration",
    legend=:bottomright,
)
add_to_plot!(dual_gap_iter_plt, 1:length(log_gd.iter_time), log_gd.dual_gap, "Gradient", :coral)
add_to_plot!(dual_gap_iter_plt, 1:length(log_sketch.iter_time), log_sketch.dual_gap, "Sketch", :purple)
add_to_plot!(dual_gap_iter_plt, 1:length(log_exact.iter_time), log_exact.dual_gap, "ADMM (exact)", :red)
add_to_plot!(dual_gap_iter_plt, 1:length(log_nys.iter_time), log_nys.dual_gap, "NysADMM", :mediumblue)
add_to_plot!(dual_gap_iter_plt, 1:length(log_nys_summable.iter_time), log_nys_summable.dual_gap, L"NysADMM $(1/t^2)$", :green)
savefig(dual_gap_iter_plt, joinpath(FIGS_PATH, "logistic-dual-gap.pdf"))

rp_iter_plt = plot(; 
    dpi=300,
    legendfontsize=14,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Primal Residual $\ell_2$ Norm",
    xlabel="Iteration",
    legend=:bottomright,
)
add_to_plot!(rp_iter_plt, 1:length(log_gd.iter_time), log_gd.rp, "Gradient", :coral)
add_to_plot!(rp_iter_plt, 1:length(log_sketch.iter_time), log_sketch.rp, "Sketch", :purple)
add_to_plot!(rp_iter_plt, 1:length(log_exact.iter_time), log_exact.rp, "ADMM (exact)", :red)
add_to_plot!(rp_iter_plt, 1:length(log_nys.iter_time), log_nys.rp, "NysADMM", :mediumblue)
add_to_plot!(rp_iter_plt, 1:length(log_nys_summable.iter_time), log_nys_summable.rp, L"NysADMM $(1/t^2)$", :green)
savefig(rp_iter_plt, joinpath(FIGS_PATH, "logistic-rp.pdf"))

rd_iter_plt = plot(; 
    dpi=300,
    legendfontsize=14,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Dual Residual $\ell_2$ Norm",
    xlabel="Iteration",
    legend=:topright,
)
add_to_plot!(rd_iter_plt, 1:length(log_gd.iter_time), log_gd.rd, "Gradient", :coral)
add_to_plot!(rd_iter_plt, 1:length(log_sketch.iter_time), log_sketch.rd, "Sketch", :purple)
add_to_plot!(rd_iter_plt, 1:length(log_exact.iter_time), log_exact.rd, "ADMM (exact)", :red)
add_to_plot!(rd_iter_plt, 1:length(log_nys.iter_time), log_nys.rd, "NysADMM", :mediumblue)
add_to_plot!(rd_iter_plt, 1:length(log_nys_summable.iter_time), log_nys_summable.rd, L"NysADMM $(1/t^2)$", :green)
savefig(rd_iter_plt, joinpath(FIGS_PATH, "logistic-rd.pdf"))

obj_val_iter_plt = plot(; 
    dpi=300,
    legendfontsize=14,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"$(p-p^\star)/p^\star$",
    xlabel="Iteration",
    legend=:bottomright,
)
add_to_plot!(obj_val_iter_plt, 1:length(log_gd.iter_time), (log_gd.obj_val .- pstar)./pstar, "Gradient", :coral)
add_to_plot!(obj_val_iter_plt, 1:length(log_sketch.iter_time), (log_sketch.obj_val .- pstar)./pstar, "Sketch", :purple)
add_to_plot!(obj_val_iter_plt, 1:length(log_exact.iter_time), (log_exact.obj_val .- pstar)./pstar, "ADMM (exact)", :red)
add_to_plot!(obj_val_iter_plt, 1:length(log_nys.iter_time), (log_nys.obj_val .- pstar)./pstar, "NysADMM", :mediumblue)
add_to_plot!(obj_val_iter_plt, 1:length(log_nys_summable.iter_time), (log_nys_summable.obj_val .- pstar)./pstar, L"NysADMM $(1/t^2)$", :green)
savefig(obj_val_iter_plt, joinpath(FIGS_PATH, "logistic-obj-val.pdf"))

logistic_plt = plot(; 
    dpi=300,
    # title="Convergence (Logistic Regression)",
    yaxis=:log,
    xlabel="Iteration",
    legend=:topright,
    ylims=(1e-10, 1000),
    legendfontsize=14,
    labelfontsize=14,
)
add_to_plot!(logistic_plt, 1:length(log_opt.iter_time), log_opt.rp, "Primal Residual", :indigo)
add_to_plot!(logistic_plt, 1:length(log_opt.iter_time), log_opt.rd, "Dual Residual", :red)
add_to_plot!(logistic_plt, 1:length(log_opt.iter_time), log_opt.dual_gap, "Duality Gap", :mediumblue)
add_to_plot!(logistic_plt, 1:length(log_opt.iter_time), sqrt(eps())*ones(length(log_opt.iter_time)), L"\sqrt{\texttt{eps}}", :black; style=:dash, lw=1)
savefig(logistic_plt, joinpath(FIGS_PATH, "logistic.pdf"))


## Divergence plot
divergence_plt = plot(; 
    dpi=300,
    yaxis=:log,
    xlabel="Iteration",
    legend=:bottomleft,
    legendfontsize=10,
    labelfontsize=14,
    ylims=(1e-2, 5e2)
)
add_to_plot!(divergence_plt, 1:length(log_sketch.iter_time), log_sketch.rp, "Primal Residual", :indigo, lw=2)
add_to_plot!(divergence_plt, 1:length(log_sketch.iter_time), log_sketch.rd, "Dual Residual", :red, lw=2)
add_to_plot!(divergence_plt, 1:length(log_sketch.iter_time), log_sketch.dual_gap, "Duality Gap", :mediumblue, lw=2)
add_to_plot!(divergence_plt, 1:length(log_sketch_no_correction.iter_time), log_sketch_no_correction.rp, nothing, :indigo, lw=2, style=:dash)
add_to_plot!(divergence_plt, 1:length(log_sketch_no_correction.iter_time), log_sketch_no_correction.rd, nothing, :red, lw=2, style=:dash)
add_to_plot!(divergence_plt, 1:length(log_sketch_no_correction.iter_time), log_sketch_no_correction.dual_gap, nothing, :mediumblue, lw=2, style=:dash)
savefig(divergence_plt, joinpath(FIGS_PATH, "logistic-divergence.pdf"))


## Some time Plots (not in paper)
obj_val_iter_plt = plot(; 
    dpi=300,
    legendfontsize=14,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"$(p-p^\star)/p^\star$",
    xlabel="Time (s)",
    # xaxis=:log,
    legend=:topright,
    # xlims=(1, 1e3),
)
add_to_plot!(obj_val_iter_plt, log_gd.iter_time .- log_gd.iter_time[1] .+ log_gd.setup_time, (log_gd.obj_val .- pstar)./pstar, "Gradient", :coral)
add_to_plot!(obj_val_iter_plt, log_sketch.iter_time .- log_sketch.iter_time[1] .+ log_sketch.setup_time, (log_sketch.obj_val .- pstar)./pstar, "Sketch", :purple)
add_to_plot!(obj_val_iter_plt, log_exact.iter_time .- log_exact.iter_time[1] .+ log_exact.setup_time, (log_exact.obj_val .- pstar)./pstar, "ADMM (exact)", :red)
add_to_plot!(obj_val_iter_plt, log_nys.iter_time .- log_nys.iter_time[1] .+ log_nys.setup_time, (log_nys.obj_val .- pstar)./pstar, "NysADMM", :mediumblue)

dual_gap_iter_plt = plot(; 
    dpi=300,
    legendfontsize=14,
    labelfontsize=14,
    yaxis=:log,
    ylabel="Dual Gap",
    xlabel="Iteration",
    legend=:bottomright,
    xaxis=:log,
)
add_to_plot!(dual_gap_iter_plt, log_gd.iter_time, log_gd.dual_gap, "Gradient", :coral)
add_to_plot!(dual_gap_iter_plt, log_sketch.iter_time, log_sketch.dual_gap, "Sketch", :purple)
add_to_plot!(dual_gap_iter_plt, log_exact.iter_time, log_exact.dual_gap, "ADMM (exact)", :red)
add_to_plot!(dual_gap_iter_plt, log_nys.iter_time, log_nys.dual_gap, "NysADMM", :mediumblue)

time_exact = log_exact.iter_time .- log_exact.iter_time[1] .+ 1.0
time_nys = log_nys.iter_time .- log_nys.iter_time[1] .+ 1.0
timing_plt = plot(; 
    dpi=300,
    legendfontsize=14,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"$(p-p^\star)/p^\star$",
    xlabel="Time (s)",
    legend=:topright,
    xaxis=:log,
    xlims=(1, 1e3),
)
add_to_plot!(timing_plt, time_exact, (log_exact.obj_val .- pstar)./pstar, "ADMM (exact)", :red)
add_to_plot!(timing_plt, time_nys,  (log_nys.obj_val .- pstar)./pstar, "NysADMM", :mediumblue)
savefig(timing_plt, joinpath(FIGS_PATH, "logistic-timing-log.pdf"))

logistic_plt = plot(; 
    dpi=300,
    # title="Convergence (Logistic Regression)",
    yaxis=:log,
    xlabel="Time (s)",
    legend=:topright,
    ylims=(1e-10, 1000),
    legendfontsize=14,
    labelfontsize=14,
)
add_to_plot!(logistic_plt, log_opt.iter_time, log_opt.rp, "Primal Residual", :indigo)
add_to_plot!(logistic_plt, log_opt.iter_time, log_opt.rd, "Dual Residual", :red)
add_to_plot!(logistic_plt, log_opt.iter_time, log_opt.dual_gap, "Duality Gap", :mediumblue)
add_to_plot!(logistic_plt, log_opt.iter_time, sqrt(eps())*ones(length(log_opt.iter_time)), L"\sqrt{\texttt{eps}}", :black; style=:dash, lw=1)
