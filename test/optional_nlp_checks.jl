using Test

order = only(ARGS)
if order == "nlp_first"
    using MadNLP, NLPModels
end
using GradPower

function nlp_test_system()
    ex = joinpath(pkgdir(GradPower), "examples")
    sys = from_psse(joinpath(ex, "2bus.raw"), joinpath(ex, "2bus_IEESGO.dyr"))
    build_network!(sys)
    runpf!(sys; verbose=false)
    dp = DynamicProblem(sys)
    initialize_dynamics!(dp, sys)
    add_event!(sys, ContingencyEvent(2, 0.2, 0.025, 0.05))
    return sys, dp
end

@testset "NLP extension: $order" begin
    if order != "nlp_first"
        loaded = Set(nameof(m) for m in Base.loaded_modules_array())
        for name in (:MadNLP, :NLPModels, :Match, :NPZ, :Plots, :Makie, :CairoMakie)
            @test !(name in loaded)
        end
        @test Base.get_extension(GradPower, :GradPowerNLPExt) === nothing
    end

    sys, dp = nlp_test_system()
    lvar, uvar = 0.8 .* dp.pvec, 1.2 .* dp.pvec
    if order != "nlp_first"
        @test_throws ArgumentError GradPower.DynamicNLP(sys, dp, 0.1, lvar, uvar)
    end
    if order == "core"
        t, traj = integrate!(dp, sys, 0.1)
        @test size(traj, 2) == length(t)
        @test all(isfinite, traj)
        @test !isdefined(GradPower, :newton)
        @test !isdefined(GradPower, :find_outliers)
        @test !isdefined(GradPower, :compare_matrix)
    end
end

if order == "gradpower_first"
    using NLPModels
    @test Base.get_extension(GradPower, :GradPowerNLPExt) === nothing
    sys, dp = nlp_test_system()
    @test_throws ArgumentError GradPower.DynamicNLP(sys, dp, 0.1, dp.pvec, dp.pvec)
    using MadNLP
end

if order != "core"
    @testset "NLP objective: $order" begin
        sys, dp = nlp_test_system()
        lvar, uvar = 0.8 .* dp.pvec, 1.2 .* dp.pvec
        ext = Base.get_extension(GradPower, :GradPowerNLPExt)
        @test ext !== nothing
        counters = NLPModels.Counters()
        nlp = GradPower.DynamicNLP(sys, dp, 0.1, lvar, uvar; counters)
        @test nlp isa NLPModels.AbstractNLPModel
        @test nlp isa ext.DynamicNLP
        @test nlp.counters === counters
        @test nlp.meta.nvar == length(dp.pvec)
        @test nlp.meta.ncon == 0
        @test nlp.meta.lvar == lvar
        @test nlp.meta.uvar == uvar

        z0, u0, p0 = copy(dp.zvec), copy(dp.uvec), copy(dp.pvec)
        p = copy(p0)
        gen = only(filter(d -> d.dtype isa GradPower.Genrou, sys.dynamic.devices))
        hidx = gen.par_ptr + findfirst(==("H"), get_param_names(gen.dtype)) - 1
        for factor in (1.0, 1.1)
            p[hidx] = factor * p0[hidx]
            reference = DynamicProblem(sys)
            initialize_dynamics!(reference, sys)
            copyto!(reference.pvec, p)
            t, traj = integrate!(reference, sys, 0.1)
            expected = sum((t[k+1] - t[k]) *
                           GradPower.functional(view(traj, :, k+1), reference.uvec, p, sys)
                           for k in 1:length(t)-1)
            @test NLPModels.obj(nlp, p) ≈ expected rtol=1e-12
            @test NLPModels.obj(nlp, p) ≈ expected rtol=1e-12
        end
        @test dp.zvec == z0
        @test dp.uvec == u0
        @test dp.pvec == p0
        @test all(!ev.status for ev in sys.dynamic.events)
        @test_throws ErrorException NLPModels.grad!(nlp, p0, similar(p0))
        jac = zeros(0, length(p0))
        @test MadNLP.jac_dense!(nlp, p0, jac) === nothing
        @test size(jac) == (0, length(p0))
    end
end
