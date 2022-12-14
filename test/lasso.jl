@testset "Lasso" begin
    Random.seed!(1)
    m, n = 100, 200
    A = sprandn(m, n, 0.7)
    @views [normalize!(A[:, i]) for i in 1:n]

    xstar = zeros(n)
    inds = randperm(n)[1:n÷10]
    xstar[inds] .= randn(length(inds))
    b = A*xstar + 1e-3*randn(m)
    γmax = norm(A'*b, Inf)
    γ = 0.1*γmax

    # Create solver
    prob = GeNIADMM.LassoSolver(A, b, γ; ρ=1.0)

    @testset "Direct" begin
        res = GeNIADMM.solve!(
            prob; indirect=false, relax=false, max_iters=200, tol=1e-4, 
            precondition=false, verbose=false
        )
        
        @test GeNIADMM.test_optimality_conditions(prob)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3

        GeNIADMM.reset!(prob)
        res = GeNIADMM.solve!(
            prob; indirect=false, relax=true, max_iters=200, tol=1e-4, 
            precondition=false, verbose=false
        )
        @test GeNIADMM.test_optimality_conditions(prob)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3
    end

    @testset "Indirect CG" begin
        GeNIADMM.reset!(prob)
        res = GeNIADMM.solve!(
            prob; indirect=true, relax=false, max_iters=200, tol=1e-4, 
            precondition=false, verbose=false
        )
        
        @test GeNIADMM.test_optimality_conditions(prob)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3

        GeNIADMM.reset!(prob)
        res = GeNIADMM.solve!(
            prob; indirect=true, relax=true, max_iters=200, tol=1e-4, 
            precondition=false, verbose=false
        )
        @test GeNIADMM.test_optimality_conditions(prob)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3
    end

    @testset "Indirect PCG" begin
        GeNIADMM.reset!(prob)
        res = GeNIADMM.solve!(
            prob; indirect=true, relax=false, max_iters=200, tol=1e-4, 
            precondition=true, verbose=false
        )
        
        @test GeNIADMM.test_optimality_conditions(prob)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3

        GeNIADMM.reset!(prob)
        res = GeNIADMM.solve!(
            prob; indirect=true, relax=true, max_iters=200, tol=1e-4, 
            precondition=true, verbose=false
        )
        @test GeNIADMM.test_optimality_conditions(prob)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3
    end

    @testset "Settings" begin
        GeNIADMM.reset!(prob)
        res = GeNIADMM.solve!(
            prob; indirect=false, relax=true, max_iters=200, tol=1e-4, 
            precondition=false, verbose=false, rho_update_iter=20
        )
        
        @test length(res.log.dual_gap) > 20
        @test GeNIADMM.test_optimality_conditions(prob)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3

        GeNIADMM.reset!(prob)
        res = GeNIADMM.solve!(
            prob; indirect=true, relax=true, max_iters=200, tol=1e-4, 
            precondition=false, verbose=false, multithreaded=true
        )
        @test GeNIADMM.test_optimality_conditions(prob)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3

        GeNIADMM.reset!(prob)
        res = GeNIADMM.solve!(
            prob; indirect=false, relax=true, max_iters=200, tol=1e-4, 
            precondition=false, verbose=false, logging=false
        )
        @test GeNIADMM.test_optimality_conditions(prob)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3
    end

    Random.seed!(1)
    m, n = 100, 200
    A = randn(m, n)
    @views [normalize!(A[:, i]) for i in 1:n]

    xstar = zeros(n)
    inds = randperm(n)[1:n÷10]
    xstar[inds] .= randn(length(inds))
    b = A*xstar + 1e-3*randn(m)
    γmax = norm(A'*b, Inf)
    γ = 0.1*γmax

    # Create solver
    prob = GeNIADMM.LassoSolver(A, b, γ; ρ=1.0)
    
    @testset "Dense" begin
        res = GeNIADMM.solve!(
            prob; indirect=true, relax=true, max_iters=500, tol=1e-4, 
            precondition=true, verbose=false
        )
        
        @test GeNIADMM.test_optimality_conditions(prob; tol=1e-3)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3

        GeNIADMM.reset!(prob)
        res = GeNIADMM.solve!(
            prob; indirect=false, relax=true, max_iters=500, tol=1e-4, 
            precondition=false, verbose=false
        )
        @test GeNIADMM.test_optimality_conditions(prob; tol=1e-3)
        @test prob.dual_gap < 1e-4
        @test prob.rp_norm < 1e-3
        @test prob.rd_norm < 1e-3
    end
end