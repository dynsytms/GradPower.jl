# Unit tests for BatchedLayout (phase 14a D4).

@testset "BatchedLayout constructor" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    M = 4
    bl = GradPower.BatchedLayout(dp, ps, M)

    @test bl.M == M
    @test bl.sys_dim == length(dp.zvec)
    @test size(bl.z) == (M, bl.sys_dim)
    @test size(bl.p) == (M, length(dp.pvec))
    @test size(bl.u) == (M, length(dp.uvec))
    @test bl.J_colptr[end] - 1 == length(bl.J_rowval)   # shared CSC pattern

    # Each scenario matches single-scenario IC
    for m in 1:M
        @test bl.z[m, :] == dp.zvec
        @test bl.p[m, :] == dp.pvec
        @test bl.u[m, :] == dp.uvec
    end
end

@testset "integrate_batched! matches integrate! per scenario" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    z0 = copy(dp.zvec); p0 = copy(dp.pvec)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(4, 0.02, 0.1, 0.15))

    # Scenarios differ in load: scale every ZIPLoad's constant-power P.
    M = 3
    zl = ps.dynamic.layout.zipload
    function make_bl()
        dp.zvec .= z0; dp.pvec .= p0   # integrate! overwrites dp.zvec
        bl = GradPower.BatchedLayout(dp, ps, M)
        for m in 1:M, k in 1:zl.n
            bl.p[m, Int(zl.par_ptr[k])] *= 1 + 0.05*(m - 1)
        end
        return bl
    end
    for (reuse, pred) in ((true, true), (false, false))
        bl = make_bl()
        tvec, trajs = GradPower.integrate_batched!(bl, ps, 0.5;
            jacobian_reuse=reuse, predictor=pred)
        for m in 1:M
            dp.zvec .= z0; dp.pvec .= bl.p[m, :]
            tv, ref = GradPower.integrate!(dp, ps, 0.5;
                jacobian_reuse=reuse, predictor=pred)
            @test tvec == tv
            # Same algorithm, same inputs: identical trajectories.
            @test trajs[m] == ref
            @test bl.z[m, :] == ref[:, end]
        end
        @test trajs[1] != trajs[M]   # the scenarios really differ
        # Threaded and serial runs agree bit for bit.
        _, trajs_ser = GradPower.integrate_batched!(make_bl(), ps, 0.5;
            jacobian_reuse=reuse, predictor=pred, threaded=false)
        @test all(trajs[m] == trajs_ser[m] for m in 1:M)
    end
    dp.zvec .= z0; dp.pvec .= p0
    for ev in ps.dynamic.events; GradPower.deactivate!(ev); end
    empty!(ps.dynamic.events)
end
