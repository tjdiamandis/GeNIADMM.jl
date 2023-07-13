# resets the solver
function reset!(solver::MLSolver{T}) where {T}
    n = solver.data.n
    solver.obj_val = zero(T)
    solver.loss = zero(T)
    solver.dual_gap = zero(T)
    solver.xk .= zeros(T, n)
    solver.x̃k .= zeros(T, n)
    solver.zk .= zeros(T, n)
    solver.yk .= zeros(T, n)
    solver.ρ = 1.0
    solver.lhs_op.ρ[1] = 1.0
    solver.α = 1.5
    return nothing
end


function NystromSketch_ATA_logistic(A::AbstractMatrix{T}, r::Int, solver::LogisticSolver) where {T}
    m, n = size(A)
    Y = zeros(T, n, r)
    cache = zeros(m, r)
    
    Ω = 1/sqrt(n) * randn(n, r)
    mul!(cache, A, Ω)
    @. cache *= solver.lhs_op.wk
    mul!(Y, A', cache)

    ν = sqrt(n)*eps(norm(Y))
    @. Y = Y + ν*Ω

    Z = zeros(r, r)
    mul!(Z, Ω', Y)
    Z[diagind(Z)] .+= ν                 # for numerical stability

    B = Y / cholesky(Symmetric(Z)).U
    U, Σ, _ = svd(B)
    Λ = Diagonal(max.(0, Σ.^2 .- ν))

    return NystromSketch(U, Λ)
end


function build_preconditioner(solver::LassoSolver, r0; μ)
    ATA_nys = RP.NystromSketch_ATA(solver.lhs_op.A, r0, r0)
    return RP.NystromPreconditionerInverse(ATA_nys, solver.ρ + solver.μ)
end


function build_preconditioner(solver::LogisticSolver, r0; μ)
    ATA_nys = NystromSketch_ATA_logistic(solver.lhs_op.A, r0, solver)
    return RP.NystromPreconditionerInverse(ATA_nys, solver.ρ)
end


function compute_KKT_mat(solver::LassoSolver)
    return Symmetric(solver.lhs_op.A'*solver.lhs_op.A + (solver.ρ + solver.μ) * I)
end


function update_wk!(solver::LogisticSolver)
    vm = solver.vm
    mul!(vm, solver.lhs_op.A, solver.x̃k)
    @. vm *= solver.data.b

    # qᵏ = exp(vm) / (1 + exp(vm))
    @. solver.lhs_op.wk = logistic(vm) / (1 + exp(vm))
    return nothing
end


function compute_KKT_mat(solver::LogisticSolver)
    update_wk!(solver)
    return Symmetric(solver.lhs_op.A'*Diagonal(solver.lhs_op.wk)*solver.lhs_op.A + solver.ρ * I)
end


function compute_rhs!(solver::LassoSolver)
    @. solver.rhs = solver.data.ATb + solver.ρ * (solver.zk - solver.yk)
    return nothing
end


function compute_rhs!(solver::LogisticSolver)
    # compute wᵏ & qᵏ
    # vm = Ax ⟹ vmᵢ = bᵢãᵢᵀx
    vm = solver.vm
    mul!(vm, solver.lhs_op.A, solver.x̃k)
    @. vm *= solver.data.b

    # qᵏ = exp(vm) / (1 + exp(vm))
    @. solver.qk = logistic(vm)
    # @. solver.qk = exp(vm) / (1 + exp(vm))

    # wᵏ = exp(vm) / (1 + exp(vm))² = qᵏ / (1 + exp(vm))
    @. solver.lhs_op.wk = solver.qk / (1 + exp(vm))

    # rhs = Aᵀ(qᵏ + diag(wᵏ)Ax) + ρ(zᵏ - yᵏ)
    @. vm = -solver.qk + solver.lhs_op.wk * vm
    @. vm *= solver.data.b
    mul!(solver.vn, solver.lhs_op.A', vm)
    @. solver.rhs = solver.vn + solver.ρ * (solver.zk - solver.yk)
    return nothing
end


# Gradient Descent ADMM
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#   Note that this formulation is equivalent to that in the paper with
#   ηᵏ = (1 - ηρ)/η, where η = 1 / (1.1λmax(AᵀA) + ρ),
#   ⟹ ηᵏ = 1.1λmax(AᵀA)
function update_x̃_gd!(
    solver::LassoSolver;
    P=I,
    logging=false
)
    compute_rhs!(solver)

    # if logging, time the GD step
    if logging
        time_start = time_ns()
    end

    # P = λmax(AᵀA)
    η = 1 / (1.1P + solver.ρ + solver.μ)
    mul!(solver.Az, solver.lhs_op.A, solver.x̃k)
    mul!(solver.ATAz, solver.lhs_op.A', solver.Az)
    @. solver.x̃k = solver.x̃k - η * (solver.ATAz + (solver.ρ + solver.μ)*solver.x̃k - solver.rhs)
    
    if logging 
        return (time_ns() - time_start) / 1e9
    else
        return nothing
    end
end


function update_x̃_gd!(
    solver::LogisticSolver;
    P=I,
    logging=false
)
    compute_rhs!(solver)

    # if logging, time the GD step
    if logging
        time_start = time_ns()
    end

    # P = λmax(AᵀA) ≥ λmax(Aᵀdiag(wᵏ)A) since wᵏ ≤ 1
    η = 1 / (1.1P + solver.ρ)
    
    # vm = Ax ⟹ vmᵢ = bᵢãᵢᵀx
    vm = solver.vm
    mul!(vm, solver.lhs_op.A, solver.x̃k)
    @. vm *= solver.data.b

    # qᵏ = exp(vm) / (1 + exp(vm))
    @. solver.qk = logistic(vm)
    @. solver.qk *= solver.data.b
    mul!(solver.vn, solver.lhs_op.A', solver.qk)
    @. solver.x̃k = solver.x̃k - η * (solver.vn + solver.ρ*(solver.x̃k - solver.zk + solver.yk))
    
    if logging 
        return (time_ns() - time_start) / 1e9
    else
        return nothing
    end
end


# Sketch and Solve ADMM
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
function update_x̃_sketch!(
    solver::LassoSolver{T},
    Enorm;
    P=I,
    logging=false,
    r0=nothing
) where {T}
    # RHS_sketch = RHS + Âx - AᵀAx + σIx
    compute_rhs!(solver)
    mul!(solver.Az, solver.lhs_op.A, solver.x̃k)
    mul!(solver.ATAz, solver.lhs_op.A', solver.Az)
    mul!(solver.vn, P, solver.x̃k)
    @. solver.rhs += solver.vn - solver.ATAz + Enorm * solver.x̃k

    # if logging, time the GD step
    if logging
        time_start = time_ns()
    end

    # P = Ânys
    τ = Enorm + solver.ρ + solver.μ
    r = length(P.Λ.diag)
    @views mul!(solver.vn[1:r], P.U', solver.rhs)
    @. @views solver.vn[1:r] *= one(T)/(P.Λ.diag + τ) - one(T)/τ
    @views mul!(solver.x̃k, P.U, solver.vn[1:r])
    @. solver.x̃k += one(T)/τ * solver.rhs

    if logging 
        return (time_ns() - time_start) / 1e9
    else
        return nothing
    end
end


function update_x̃_sketch!(
    solver::LogisticSolver{T},
    Enorm;
    P=I,
    logging=false,
    r0=50,
) where {T}
    # RHS_sketch = RHS + Âx - AᵀAx + σIx
    solver.rhs = solver.ρ * (solver.zk - solver.yk)
    
    # vm = Ax ⟹ vmᵢ = bᵢãᵢᵀx
    vm = solver.vm
    mul!(vm, solver.lhs_op.A, solver.x̃k)
    @. vm *= solver.data.b

    # qᵏ = exp(vm) / (1 + exp(vm))
    @. solver.qk = logistic(vm)
    @. solver.lhs_op.wk = solver.qk / (1 + exp(vm))

    @. solver.qk *= solver.data.b
    mul!(solver.vn, solver.lhs_op.A', solver.qk)
    @. solver.rhs -= solver.vn

    mul!(solver.vn, P, solver.x̃k)
    solver.rhs += solver.vn + Enorm * solver.x̃k

    # if logging, time the GD step
    if logging
        time_start = time_ns()
    end

    # P = Ânys
    τ = Enorm + solver.ρ
    r = length(P.Λ.diag)
    @views mul!(solver.vn[1:r], P.U', solver.rhs)
    @. @views solver.vn[1:r] *= one(T)/(P.Λ.diag + τ) - one(T)/τ
    @views mul!(solver.x̃k, P.U, solver.vn[1:r])
    @. solver.x̃k += one(T)/τ * solver.rhs

    if logging 
        return (time_ns() - time_start) / 1e9
    else
        return nothing
    end
end


# NysADMM
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
function update_x̃!(
    solver::MLSolver,
    linsys_solver::CgSolver;
    P=I,
    logging=false,
    linsys_tol=nothing
)
    compute_rhs!(solver)

    # if logging, time the linear system solve
    if logging
        time_start = time_ns()
    end

    if isnothing(linsys_tol)
        linsys_tol = min(sqrt(solver.rp_norm * solver.rd_norm), 1e-1)
    end

    # warm start if past first iteration
    !isinf(solver.rp_norm) && warm_start!(linsys_solver, solver.x̃k)
    cg!(
        linsys_solver, solver.lhs_op, solver.rhs;
        M=P, rtol=linsys_tol
    )
    !issolved(linsys_solver) && error("CG failed")
    solver.x̃k .= linsys_solver.x

    if logging 
        return (time_ns() - time_start) / 1e9
    else
        return nothing
    end
end


function cholesky_update(linsys_solver, ::LassoSolver)
    return linsys_solver
end


function cholesky_update(::Cholesky, solver::LogisticSolver)
    A = solver.lhs_op.A
    return cholesky(Symmetric(A'*Diagonal(solver.lhs_op.wk)*A) + solver.ρ * I)
end


function cholesky_update(linsys_solver::SuiteSparse.CHOLMOD.Factor, solver::LogisticSolver)
    A = solver.lhs_op.A
    return cholesky!(
        linsys_solver,
        Symmetric(A'*Diagonal(solver.lhs_op.wk)*A) + solver.ρ * I
    )
end


# Exact ADMM
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
function update_x̃!(
    solver::MLSolver,
    linsys_solver::Union{Cholesky, SuiteSparse.CHOLMOD.Factor};
    P=I, 
    logging=false,
    linsys_tol=nothing,
)
    compute_rhs!(solver)
    
    # if logging, time the linear system solve
    if logging
        time_start = time_ns()
    end

    linsys_solver = cholesky_update(linsys_solver, solver)

    # Direct solver
    if typeof(linsys_solver) <: Cholesky
        ldiv!(solver.x̃k, linsys_solver, solver.rhs)
    else
        solver.x̃k .= linsys_solver \ solver.rhs
    end

    if logging 
        return (time_ns() - time_start) / 1e9
    else
        return nothing
    end
end


# Solve subproblem with LBFGS
function update_x̃_exact!(
    solver::LogisticSolver{T};
    logging=false
) where {T}
    
    # if logging, time the linear system solve
    if logging
        time_start = time_ns()
    end

    function fg!(F, G, x, solver)
        @. solver.vn = x - solver.zk + solver.yk
        mul!(solver.Az, solver.lhs_op.A, x)
        @. solver.Az *= solver.data.b
    
        # G = Aᵀq(x) + ρ(x - zᵏ + yᵏ)
        # q(x) = logistic.(Ax)
        if !isnothing(G)
            @. solver.vm = logistic(solver.Az)
            @. solver.vm *= solver.data.b
            mul!(G, solver.lhs_op.A', solver.vm)
            @. G += solver.ρ * solver.vn
        end
        if !isnothing(F)
            return sum(w->log1pexp(w), solver.Az) + solver.ρ/2 * norm(solver.vn)^2
        end
    end
    fg!(F, G, x) = fg!(F, G, x, solver)

    res = Optim.optimize(Optim.only_fg!(fg!), solver.x̃k, method = Optim.BFGS())
    solver.x̃k .= Optim.minimizer(res)

    if logging 
        return (time_ns() - time_start) / 1e9
    else
        return nothing
    end
end


function update_x!(solver::MLSolver{T}; relax=true) where {T <: Real}
    if relax
        @. solver.xk = solver.α * solver.x̃k + (one(T) - solver.α) * solver.zk
    else
        solver.xk .= solver.x̃k
    end

    return nothing
end


function soft_threshold(x::T, y::T, κ::T) where {T <: Real}
    tmp = x + y
    return sign(tmp) * max(zero(T), abs(tmp) - κ)
end


function update_z!(solver::Union{LassoSolver, LogisticSolver})
    solver.zk .= soft_threshold.(solver.xk, solver.yk, solver.γ / solver.ρ)
end


function update_y!(solver::MLSolver{T}) where {T <: Real}
    @. solver.yk = solver.yk + solver.xk - solver.zk
end


function obj_val!(solver::LassoSolver{T}) where {T}
    z = solver.zk
    Az = solver.Az
    ATAz = solver.ATAz

    mul!(Az, solver.lhs_op.A, z)
    mul!(ATAz, solver.lhs_op.A', Az)

    # =  bᵀAz
    bTAz = dot(solver.data.ATb, z)

    solver.loss = 0.5*(dot(z, ATAz) - 2bTAz + solver.data.bTb)
    solver.obj_val = solver.loss + 0.5 * solver.μ * sum(x->x^2, z) + solver.γ * norm(z, 1)
    return solver.obj_val
end


function obj_val!(solver::LogisticSolver{T}) where {T}
    z = solver.zk
    Az = solver.Az
    b = solver.data.b
    vm = solver.vm

    mul!(Az, solver.lhs_op.A, z)
    @. vm = b * Az
    @. vm = log1pexp(vm)
    solver.loss = sum(vm)

    solver.obj_val = solver.loss + solver.γ * norm(z, 1)
    return solver.obj_val
end


# NOTE: assumes that objective has been updated already, specifically
#   1) solver.loss & solver.obj_val have been updated
#   2) solver.ATAz has been updated
function dual_gap!(solver::LassoSolver)
    bTb = solver.data.bTb
    γ = solver.γ
    sq_error = 2 * solver.loss

    # compute |AᵀAz - Aᵀb|_inf
    @. solver.vn = solver.ATAz - solver.data.ATb
    normalization = norm(solver.vn, Inf)

    # compute bᵀAz
    bTAz = dot(solver.data.ATb, solver.zk)

    # ν = γ(Ax - b) / |Ax - b|_inf
    # dual obj = -1/2*νᵀν - νᵀb
    dual_obj = -0.5*γ^2/normalization^2*sq_error - γ/normalization * (bTAz - bTb)

    solver.dual_gap = (solver.obj_val - dual_obj) / min(solver.obj_val, abs(dual_obj))
    return solver.dual_gap
end


function dual_gap!(solver::LogisticSolver{T}) where {T}
    @. solver.vm = solver.data.b * solver.Az
    @. solver.vm = logistic(solver.vm)

    @. solver.vm *= solver.data.b
    mul!(solver.vn, solver.lhs_op.A', solver.vm)
    normalization = norm(solver.vn, Inf)
    normalization = solver.γ / normalization
    
    @. solver.vm *= solver.data.b
    @. solver.vm *= normalization

    
    # Note: νᵀb = 0
    ν = solver.vm
    @inline fconj(w::T) = w > 0 && w < 1 ? (one(T) - w) * log(one(T) - w) + w * log(w) : Inf
    # ν .= clip.(ν, 1e-5, 1.0-1e-5)
    ν .= fconj.(ν)
    dual_obj = -sum(ν)
    # dual_obj = sum(x->(x > 0 && x < 1) ? (x - one(T))*log(one(T)-x) - x*log(x) : Inf, solver.vm)

    if isinf(dual_obj)
        solver.dual_gap = Inf
    else
        solver.dual_gap = (solver.obj_val - dual_obj) / min(solver.obj_val, abs(dual_obj))
    end
end


function compute_residuals!(solver::MLSolver)
    @. solver.rp = solver.xk - solver.zk
    @. solver.rd = solver.ρ * (solver.zk - solver.zk_old)
    
    solver.rp_norm = norm(solver.rp)
    solver.rd_norm = norm(solver.rd)
    return nothing
end


function update_ρ!(solver::MLSolver; update_buffer=10, rho_update_factor=2)
    # NOTE: using scaled ADMM, so must rescale variable after ρ update
    if solver.rp > update_buffer * solver.rd
        solver.ρ = solver.ρ * rho_update_factor
        solver.lhs_op.ρ[1] = solver.ρ
        @. solver.yk *= 1 / rho_update_factor
        return true
    elseif solver.rd > update_buffer * solver.rp
        solver.ρ = solver.ρ / rho_update_factor
        solver.lhs_op.ρ[1] = solver.ρ
        @. solver.yk *= rho_update_factor
        return true
    end

    return false
end

# Dual gap criterion
function converged(solver::LassoSolver, tol)
    return iszero(solver.μ) ? solver.dual_gap ≤ tol : solver.rp_norm < tol && solver.rd_norm < tol
end

# Relative KKT residual criterion
# function converged(solver::LassoSolver{T}, tol) where {T}
#     @. solver.vn = solver.zk - solver.ATAz + solver.data.ATb
#     @. solver.vn = soft_threshold(solver.vn, zero(T), solver.γ)
#     @. solver.vn = solver.zk - solver.vn
#     num = norm(solver.vn)
#     denom = 1 + norm(solver.zk) + sqrt(2 * solver.loss)
#     relative_KKT = num / denom
#     @show relative_KKT
#     return relative_KKT ≤ tol
# end

function converged(solver::LogisticSolver, tol)
    return solver.dual_gap ≤ tol
    # return solver.rp_norm < tol && solver.rd_norm < tol
end


function solve!(
    solver::MLSolver;
    relax::Bool=true,
    logging::Bool=true,
    indirect::Bool=false,
    precondition::Bool=true,
    tol=1e-4,
    max_iters::Int=100,
    max_time_sec::Real=1200.0,
    print_iter::Int=25,
    rho_update_iter::Int=50,
    sketch_update_iter::Int=20,
    verbose::Bool=true,
    multithreaded::Bool=false,
    gd_x_update::Bool=false,
    sketch_solve_x_update::Bool=false,
    sketch_rank=10,
    logistic_exact_solve::Bool=false,
    sketch_solve_update_iter=20,
    summable_step_size::Bool=false,
    add_Enorm::Bool=true
)
    !indirect && precondition && ArgumentError("Cannot precondition direct solve")

    setup_time_start = time_ns()
    verbose && @printf("Starting setup...")

    # --- parameters & data ---
    m, n = solver.data.m, solver.data.n

    # --- setup ---
    t = 1
    solver.dual_gap = Inf
    solver.obj_val = Inf
    solver.loss = Inf
    solver.rp_norm = Inf
    solver.rd_norm = Inf
    solver.xk .= zeros(n)
    solver.x̃k .= zeros(n)
    solver.zk .= zeros(n)
    solver.yk .= zeros(n)

    # --- enable multithreaded BLAS ---
    if multithreaded
        BLAS.set_num_threads(Sys.CPU_THREADS)
    else
        BLAS.set_num_threads(1)
    end

    # --- Setup Linear System Solver ---
    precond_time = 0.0
    if indirect && !precondition
        linsys_solver = CgSolver(n, n, typeof(solver.xk))
        P = I
    elseif indirect && precondition
        linsys_solver = CgSolver(n, n, typeof(solver.xk))

        verbose && @printf("\n\tPreconditioning...")
        precond_time_start = time_ns()

        
        r0 = n ≥ 1_000 ? 50 : m ÷ 20
        P = build_preconditioner(solver, r0; μ=solver.μ)
        precond_time = (time_ns() - precond_time_start) / 1e9
        
        r = length(P.A_nys.Λ.diag)
        precond_time = (time_ns() - precond_time_start) / 1e9
        verbose && @printf("\n\t - Preconditioned (rank %d) in %6.3fs", r, precond_time)
    else
        KKT_mat = compute_KKT_mat(solver)
        linsys_solver = cholesky(KKT_mat)
        P = I
    end

    if gd_x_update
        P = RP.eigmax_power(QuadForm(solver.lhs_op.A); q=10)
    elseif sketch_solve_x_update
        r = sketch_rank
        S = RP.NystromSketch_ATA(solver.lhs_op.A, r, r)
        if add_Enorm
            Enorm = typeof(solver) <: LassoSolver ? estimate_norm_E(QuadForm(solver.lhs_op.A), S) : Inf
        else
            Enorm = 0.0
        end
    end

    # --- Logging ---
    if logging
        tmp_log = create_temp_log(max_iters)
    end

    setup_time = (time_ns() - setup_time_start) / 1e9
    verbose && @printf("\nSetup in %6.3fs\n", setup_time)

    # --- Print Headers ---
    headers = ["Iteration", "Objective", "RMSE", "Dual Gap", "r_primal", "r_dual", "ρ", "Time"]
    verbose && print_header_ml(headers)

    # --- Print 0ᵗʰ iteration ---
    obj_val!(solver)
    dual_gap!(solver)
    verbose && print_iter_func_ml((
        string(0), solver.obj_val, sqrt(solver.loss / m), solver.dual_gap, Inf,
        Inf, solver.ρ, 0.0
    ))

    # --------------------------------------------------------------------------
    # --------------------- ITERATIONS -----------------------------------------
    # --------------------------------------------------------------------------
    solve_time_start = time_ns()
    while t <= max_iters && 
        (time_ns() - solve_time_start) / 1e9 < max_time_sec &&
        !converged(solver, tol)

        # --- Update Iterates ---
        if gd_x_update
            time_linsys = update_x̃_gd!(solver; P=P, logging=logging)
        elseif sketch_solve_x_update
            if typeof(solver) <: LogisticSolver && (t == 1 || t % sketch_solve_update_iter == 0)
                #  -- Update wk --
                # vm = Ax ⟹ vmᵢ = bᵢãᵢᵀx
                vm = solver.vm
                mul!(vm, solver.lhs_op.A, solver.x̃k)
                @. vm *= solver.data.b

                # qᵏ = exp(vm) / (1 + exp(vm))
                @. solver.qk = logistic(vm)
                @. solver.lhs_op.wk = solver.qk / (1 + exp(vm))
                # -- Sketch --
                S = NystromSketch_ATA_logistic(solver.lhs_op.A, r, solver)
                Enorm = add_Enorm ? estimate_norm_E(QuadForm(solver.lhs_op.A), S) : 0.0
            end
            time_linsys = update_x̃_sketch!(solver, Enorm; P=S, logging=logging, r0=sketch_rank)
        elseif logistic_exact_solve
            time_linsys = update_x̃_exact!(solver; logging=logging)
        else
            linsys_tol = summable_step_size ? 1.0/t^2 : nothing
            time_linsys = update_x̃!(solver, linsys_solver; P=P, logging=logging, linsys_tol=linsys_tol)
        end
        # t == 2 && error()
        update_x!(solver; relax=relax)
        solver.zk_old .= solver.zk
        update_z!(solver)
        update_y!(solver)


        # --- Update ρ ---
        compute_residuals!(solver)
        if t % rho_update_iter == 0
            ρ_old = solver.ρ
            updated_rho = update_ρ!(solver)
            
            if updated_rho && !indirect && typeof(solver) <: LassoSolver
                # NOTE: logistic solver recomputes fact at each iteration anyway
                KKT_mat[diagind(KKT_mat)] .+= (solver.ρ .- ρ_old)
                linsys_solver = cholesky(KKT_mat)
            elseif updated_rho && precondition && !(sketch_solve_x_update || gd_x_update)
                reg_term = typeof(solver) <: LogisticSolver ? solver.ρ : solver.ρ + solver.μ
                P = RP.NystromPreconditionerInverse(P.A_nys, reg_term)
            end
        end

        # --- Update sketch ---
        if precondition && t % sketch_update_iter == 0 && typeof(solver) <: LogisticSolver
            update_wk!(solver)
            ATA_nys = NystromSketch_ATA_logistic(solver.lhs_op.A, r0, solver)
            P = RP.NystromPreconditionerInverse(ATA_nys, solver.ρ)
        end


        # --- Update objective & dual gap ---
        obj_val!(solver)
        dual_gap!(solver)

        # --- Logging ---
        time_sec = (time_ns() - solve_time_start) / 1e9
        if logging
            tmp_log.dual_gap[t] = solver.dual_gap
            tmp_log.obj_val[t] = solver.obj_val
            tmp_log.iter_time[t] = time_sec
            tmp_log.linsys_time[t] = time_linsys
            tmp_log.rp[t] = solver.rp_norm
            tmp_log.rd[t] = solver.rd_norm
        end

        # --- Printing ---
        if verbose && (t == 1 || t % print_iter == 0)
            print_iter_func_ml((
                string(t),
                solver.obj_val,
                sqrt(solver.loss / m),
                solver.dual_gap,
                solver.rp_norm,
                solver.rd_norm,
                solver.ρ,
                time_sec
            ))
        end

        t += 1
    end

    # print final iteration if havent done so
    if verbose && ((t-1) % print_iter != 0 && (t-1) != 1)
        print_iter_func_ml((
            string(t-1),
            solver.obj_val,
            sqrt(solver.loss / m),
            solver.dual_gap,
            solver.rp_norm,
            solver.rd_norm,
            solver.ρ,
            (time_ns() - solve_time_start) / 1e9
        ))
    end

    solve_time = (time_ns() - solve_time_start) / 1e9
    verbose && @printf("\nSolved in %6.3fs, %d iterations\n", solve_time, t-1)
    verbose && @printf("Total time: %6.3fs\n", setup_time + solve_time)
    verbose && print_footer_ml()


    # --- Construct Logs ---
    if logging
        log = NysADMMLog(
            tmp_log.dual_gap[1:t-1], tmp_log.obj_val[1:t-1], tmp_log.iter_time[1:t-1],
            tmp_log.linsys_time[1:t-1],
            tmp_log.rp[1:t-1], tmp_log.rd[1:t-1],
            setup_time, precond_time, solve_time
        )
    else
        log = NysADMMLog(setup_time, precond_time, solve_time)
    end


    # --- Construct Solution ---
    res = NysADMMResult(
        solver.obj_val,
        solver.loss,
        solver.zk,                  # return zk since it is sparsified
        solver.dual_gap,
        log
    )

    return res

end