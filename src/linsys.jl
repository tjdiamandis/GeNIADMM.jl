abstract type LinearOperator end

# Linear operator AᵀA + ρI for the CG solver
struct LassoLinearOperator{T} <: LinearOperator
    # TODO: maybe store this in data & use a closure for mul!?
    A::AbstractMatrix{T}
    cache::AbstractVector{T}
    ρ::MVector{1,T}
    # γ2::MVector{1,T} # for elastic net

    function LassoLinearOperator(A::AbstractMatrix{T}, ρ::T) where {T <: AbstractFloat}
        m = size(A, 1)
        return new{T}(A, zeros(T, m), SA[ρ])
        # return new{T}(A/sqrt(m), zeros(T, m), SA[ρ])
    end
end

# y = (AᵀA + ρI)x
function LinearAlgebra.mul!(y::AbstractVector{T}, M::LassoLinearOperator{T}, x::AbstractVector{T}) where {T}
    mul!(M.cache, M.A, x)
    mul!(y, M.A', M.cache)
    @. y += M.ρ[1] * x
    return nothing
end
Base.size(M::LinearOperator) = (size(M.A, 2), size(M.A, 2))
Base.eltype(M::LinearOperator) = eltype(M.A)


# Linear operator Aᵀdiag(wᵏ)A + ρI for the CG solver
struct LogisticLinearOperator{T} <: LinearOperator
    A::AbstractMatrix{T}
    cache::AbstractVector{T}
    wk::AbstractVector{T}
    ρ::MVector{1,T}

    function LogisticLinearOperator(A::AbstractMatrix{T}, ρ::T) where {T <: AbstractFloat}
        m = size(A, 1)
        return new{T}(A, zeros(T, m), zeros(T, m), SA[ρ])
    end
end

# y = (Aᵀdiag(b)diag(wᵏ)diag(b)A + ρI)x
# NOTE: diag(b)*diag(b) = I since bᵢ ∈ {±1}
function LinearAlgebra.mul!(y::AbstractVector{T}, M::LogisticLinearOperator{T}, x::AbstractVector{T}) where {T}
    mul!(M.cache, M.A, x)
    M.cache .*= M.wk
    mul!(y, M.A', M.cache)
    @. y += M.ρ[1] * x
    return nothing
end


struct QuadForm{T} <: AbstractMatrix{T}
    A::AbstractMatrix{T}
    v::AbstractVector{T}
end

function QuadForm(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    return QuadForm{T}(A, zeros(T, m))
end

function LinearAlgebra.mul!(y, Q::QuadForm, x)
    mul!(Q.v, Q.A, x)
    mul!(y, Q.A', Q.v)
    return nothing
end

Base.size(Q::QuadForm) = (size(Q.A, 2), size(Q.A, 2))


function estimate_norm_E(A, Ahat::RP.NystromSketch{T}; q=10) where {T <: Number}
    n = size(Ahat, 2)
    u, v = zeros(T, n), zeros(T, n)
    cache = (Ahat_mul=zeros(T, n), vn=zeros(T, n))
    
    randn!(u)
    normalize!(u)
    # randn!(v, n)
    # normalize!(v)
    
    Ehat = Inf
    for _ in 1:q
        # u = (A - Ahat)*v
        mul!(cache.vn, Ahat, u; cache=cache.Ahat_mul)
        mul!(v, A, u)
        @. v = v - cache.vn
        Ehat = dot(u, v)

        normalize!(v)
        u .= v
    end
    return Ehat
end