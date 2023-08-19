abstract type MLSolver{T} end

struct LassoProblemData{T}
    # A::AbstractMatrix{T}
    ATb::AbstractVector{T}          # Store for efficiency
    b::AbstractVector{T}
    bTb::T
    m::Int
    n::Int
    function LassoProblemData(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: Real}
        m, n = size(A)
        m != length(b) && error(DimensionMismatch("Invalid dimensions for A or b"))
        return new{T}(
            # A,
            A'*b,
            # 1/m * A'*b,
            b,
            sum(x->x^2, b),
            # 1/m * sum(x->x^2, b),
            m,
            n
        )
    end
end


mutable struct LassoSolver{T} <: MLSolver{T}
    data::LassoProblemData{T}
    lhs_op::LassoLinearOperator{T}
    xk::AbstractVector{T}       # var   : primal (loss)
    x̃k::AbstractVector{T}       # var   : primal (loss--for relaxation)
    zk::AbstractVector{T}       # var   : primal (reg)
    zk_old::AbstractVector{T}   # var   : dual (prev step)
    yk::AbstractVector{T}       # var   : dual
    rp::AbstractVector{T}       # resid : primal
    rd::AbstractVector{T}       # resid : dual
    rhs::AbstractVector{T}      # cache : RHS of linear system
    Az::AbstractVector{T}       # cache : Az (size m)
    ATAz::AbstractVector{T}     # cache : AᵀAz (size n)
    vn::AbstractVector{T}       # cache : v = AᵀAz - y (size n)
    obj_val::T                  # log   : 0.5*||Ax - b||² + γ|x|₁
    loss::T                     # log   : 0.5*||Ax - b||² (uses zk)
    dual_gap::T                 # log   : obj_val - g(ν)
    rp_norm::T                  # log   : norm(rp)
    rd_norm::T                  # log   : norm(rd)
    γ::T                        # param : L1 regularization weight
    ρ::T                        # param : ADMM penalty
    α::T                        # param : relaxation
    μ::T                        # param : regularizer

    function LassoSolver(
        A::AbstractMatrix{T}, 
        b::AbstractVector{T}, 
        γ::T; 
        ρ=1.0, 
        α=1.0,
        μ=0.0
    ) where {T <: AbstractFloat}
        data = LassoProblemData(A, b)
        m, n = size(A)
        return new{T}(
            data,
            LassoLinearOperator(A, ρ, μ),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zeros(T, m),
            zeros(T, n),
            zeros(T, n),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            γ,
            ρ,
            α,
            μ
        )
    end
end


struct LogisticProblemData{T}
    b::AbstractVector{T}
    m::Int
    n::Int
    function LogisticProblemData(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: Real}
        m, n = size(A)
        m != length(b) && error(DimensionMismatch("Invalid dimensions for A or b"))
        return new{T}(
            b,
            m,
            n
        )
    end
end


mutable struct LogisticSolver{T} <: MLSolver{T}
    # NOTE: wk is stored in lhs_op
    data::LogisticProblemData{T}
    lhs_op::LogisticLinearOperator{T}
    xk::AbstractVector{T}       # var   : primal (loss)
    x̃k::AbstractVector{T}       # var   : primal (loss--for relaxation)
    zk::AbstractVector{T}       # var   : primal (reg)
    zk_old::AbstractVector{T}   # var   : dual (prev step)
    yk::AbstractVector{T}       # var   : dual
    qk::AbstractVector{T}       # cache : used in the linsys (size m)
    rhs::AbstractVector{T}      # cache : RHS of linear system (size n)
    Az::AbstractVector{T}       # cache : Az (size m)
    # ATAz::AbstractVector{T}     # cache : AᵀAz (size n)
    vm::AbstractVector{T}       # cache: (size m)
    vn::AbstractVector{T}       # cache : v = AᵀAz - y (size n)
    rp::AbstractVector{T}       # resid : primal
    rd::AbstractVector{T}       # resid : dual
    obj_val::T                  # log   : 0.5*||Ax - b||² + γ|x|₁
    loss::T                     # log   : 0.5*||Ax - b||² (uses zk)
    dual_gap::T                 # log   : obj_val - g(ν)
    rp_norm::T                  # log   : norm(rp)
    rd_norm::T                  # log   : norm(rd)
    γ::T                        # param : L1 regularization weight
    ρ::T                        # param : ADMM penalty
    α::T                        # param : relaxation
    μ::T                        # param : regularizer (unused)

    function LogisticSolver(
        A::AbstractMatrix{T}, 
        b::AbstractVector{T}, 
        γ::T; 
        ρ=1.0, 
        α=1.0
    ) where {T <: AbstractFloat}
        data = LogisticProblemData(A, b)
        m, n = size(A)
        return new{T}(
            data,
            LogisticLinearOperator(A, ρ),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zeros(T, m),
            zeros(T, n),
            zeros(T, m),
            zeros(T, m),
            zeros(T, n),
            zeros(T, n),
            zeros(T, n),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            γ,
            ρ,
            α,
            zero(T)
        )
    end
end


struct NysADMMLog{T <: AbstractFloat}
    dual_gap::Union{AbstractVector{T}, Nothing}
    obj_val::Union{AbstractVector{T}, Nothing}
    iter_time::Union{AbstractVector{T}, Nothing}
    linsys_time::Union{AbstractVector{T}, Nothing}
    rp::Union{AbstractVector{T}, Nothing}
    rd::Union{AbstractVector{T}, Nothing}
    assump::Union{AbstractVector{T}, Nothing}
    setup_time::T
    precond_time::T
    solve_time::T
end
function NysADMMLog(setup_time::T, precond_time::T, solve_time::T) where {T <: AbstractFloat}
    return NysADMMLog(
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        setup_time, precond_time, solve_time
    )
end


function create_temp_log(max_iters::Int)
    return NysADMMLog(
        zeros(max_iters),
        zeros(max_iters),
        zeros(max_iters),
        zeros(max_iters),
        zeros(max_iters),
        zeros(max_iters),
        zeros(max_iters),
        0.0,
        0.0,
        0.0
    )
end



struct NysADMMResult{T}
    obj_val::T                 # primal objective
    loss::T                # 0.5*||Ax - b||²
    x::AbstractVector{T}       # primal soln
    dual_gap::T                # duality gap
    # vT::AbstractVector{T}      # dual certificate
    log::NysADMMLog{T}
end