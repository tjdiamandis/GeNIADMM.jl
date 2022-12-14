function print_header_ml(data)
    @printf(
        "\n──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
    @printf(
        "%13s %14s %14s %14s %14s %14s %14s %14s\n",
        data[1],
        data[2],
        data[3],
        data[4],
        data[5],
        data[6],
        data[7],
        data[8]
    )
    @printf(
        "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
end


function print_footer_ml()
    @printf(
        "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
    )
end


function print_iter_func_ml(data)
    @printf(
        "%13s %14.3e %14.3e %14.3e %14.3e %14.3e %14.3e %13.3f\n",
        data[1],
        Float64(data[2]),
        Float64(data[3]),
        Float64(data[4]),
        Float64(data[5]),
        Float64(data[6]),
        data[7],
        data[8]
    )
end


function test_optimality_conditions(solver::LassoSolver; tol=1e-4)
    vn = solver.vn
    γ = solver.γ
    x = solver.zk

    mul!(solver.Az, solver.lhs_op.A, x)
    @. solver.Az -= solver.data.b
    mul!(vn, solver.lhs_op.A', solver.Az)

    # Optimality conditions: Aᵀ(Ax - b) + γ∂|x|₁ ≈ 0
    return all(abs.(vn[x .== 0]) .<= γ) && 
           all(abs.(vn[x .> 0] .+ γ) .< tol) && 
           all(abs.(vn[x .< 0] .- γ) .< tol)
end


function test_optimality_conditions(solver::LogisticSolver; tol=1e-4)
    vn = solver.vn
    γ = solver.γ
    x = solver.zk

    mul!(solver.Az, solver.lhs_op.A, x)
    @. solver.vm = solver.data.b * solver.Az
    @. solver.vm = exp(solver.vm) / (1 + exp(solver.vm))
    @. solver.vm *= solver.data.b
    mul!(vn, solver.lhs_op.A', solver.vm)
    

    # Optimality conditions: Aᵀ(Ax - b) + γ∂|x|₁ ≈ 0
    return all(abs.(vn[x .== 0]) .<= γ) && 
           all(abs.(vn[x .> 0] .+ γ) .< tol) && 
           all(abs.(vn[x .< 0] .- γ) .< tol)
end