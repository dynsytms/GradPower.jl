# LoadStepEvent: relative load steps on a ZIPLoad (P, Q, G, B scaled from
# their initial values), applied between steps like the other events.

@testset "Load step events" begin
    ex = joinpath(@__DIR__, "..", "examples")
    function setup(raw = "ieee9_v33.raw", dyr = "ieee9bus_gov.dyr")
        ps = from_psse(joinpath(ex, raw), joinpath(ex, dyr))
        GradPower.build_network!(ps); GradPower.runpf!(ps)
        dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
        ps, dp, copy(dp.zvec), copy(dp.pvec)
    end
    ps, dp, z0, p0 = setup()
    dt = 1/120
    run(; tf = 0.5, kw...) = (dp.zvec .= z0; r = integrate!(dp, ps, tf; dt = dt, kw...); (r..., copy(dp.zvec)))
    tv0, tr0, _ = run()
    li = findfirst(d -> d.dtype isa GradPower.ZIPLoad, ps.dynamic.devices)
    ld = ps.dynamic.devices[li]
    bus_ext = ps.buses[ld.dtype.bus].i
    pp = ld.par_ptr

    @testset "lookup by bus + id" begin
        @test find_load_device(ps, bus_ext, ld.dtype.id) == li
        @test find_load_device(ps, bus_ext, "'$(strip(ld.dtype.id))'") == li   # quotes/blanks ignored
        @test_throws ArgumentError find_load_device(ps, bus_ext, "ZZ")
        @test_throws ArgumentError find_load_device(ps, 99999, "1")
        gi = findfirst(d -> d.dtype isa GradPower.Genrou, ps.dynamic.devices)
        @test_throws ArgumentError add_load_step_event!(ps, LoadStepEvent(gi, 0.1, 1.1))
        @test_throws ArgumentError add_load_step_event!(ps, LoadStepEvent(li, 0.1, -1.0))
        @test_throws ArgumentError add_load_step_event!(ps, LoadStepEvent(li, 0.1, NaN))
        @test isempty(ps.dynamic.load_step_events)
    end

    @testset "scale 1.0 changes nothing (beyond the event re-solve)" begin
        # Any event re-solves the algebraic equations and refactors, so the
        # result moves at Newton-tolerance level only.
        add_load_step_event!(ps, create_load_step_event(ps, bus_ext, ld.dtype.id, 0.1, 1.0))
        tv, tr, _ = run()
        @test maximum(abs.(tr .- tr0)) < 1e-8
        empty!(ps.dynamic.load_step_events)
    end

    @testset "step = run to t, scale parameters, restart" begin
        # Integrate to the step, scale p by hand, re-solve the algebraic
        # equations (dt = 0, what the event does), continue: same trajectory.
        tstep = 0.1; k = Int(round(tstep / dt))
        add_load_step_event!(ps, create_load_step_event(ps, bus_ext, ld.dtype.id, tstep, 1.2))
        tv, tr, zf = run(jacobian_reuse = false, predictor = false)
        @test dp.pvec == p0                       # restored
        empty!(ps.dynamic.load_step_events)
        _, trn, _ = run(jacobian_reuse = false, predictor = false)
        @test tr[:, 1:k+1] == trn[:, 1:k+1]        # identical up to the step
        @test maximum(abs.(tr[:, end] .- trn[:, end])) > 1e-3   # and it acts

        _, trA, _ = run(tf = tstep, jacobian_reuse = false, predictor = false)
        p1 = copy(p0); for o in (0, 1, 7, 8); p1[pp + o] *= 1.2; end
        dp.pvec .= p1
        z = trA[:, end]
        f = zeros(length(z)); J = GradPower.preallocate_jacobian(ps)
        GradPower.beuler_jac!(J, z, z, dp.uvec, dp.pvec, ps, ps.dynamic.diff_dim, 0.0)
        fact = Ref{Any}(GradPower.ds_solver(KLUBackend(), J; tuned = true))
        @test GradPower.newton_step!(z, f, J, fact, copy(z), dp.uvec, dp.pvec, ps, 0.0; tol = 1e-10)
        dp.zvec .= z
        _, trB = integrate!(dp, ps, 0.5 - tstep; dt = dt, jacobian_reuse = false, predictor = false)
        dp.pvec .= p0
        @test maximum(abs.(trB[:, end] .- tr[:, end])) < 1e-9
        @test maximum(abs.(trB[:, 2:end] .- tr[:, k+2:end])) < 1e-9
    end

    @testset "parameters during the run" begin
        # Exactly the four ZIP entries of this load are scaled from their
        # initial values, from the step on; later steps do not compound.
        add_load_step_event!(ps, create_load_step_event(ps, bus_ext, ld.dtype.id, 0.1, 1.1))
        add_load_step_event!(ps, create_load_step_event(ps, bus_ext, ld.dtype.id, 0.2, 0.5))
        snaps = Dict{Int,Vector{Float64}}()
        cb = (k, t, z) -> (snaps[k] = copy(dp.pvec))
        run(tf = 0.3, step_callback = cb)
        k1 = Int(round(0.1 / dt)); k2 = Int(round(0.2 / dt))
        idx = [pp + o for o in (0, 1, 7, 8)]; rest = setdiff(eachindex(p0), idx)
        @test snaps[k1 + 1] == p0                          # stored step k1+1 = before the step's events
        @test snaps[k1 + 2][idx] == 1.1 .* p0[idx] && snaps[k1 + 2][rest] == p0[rest]
        @test snaps[k2 + 2][idx] == 0.5 .* p0[idx] && snaps[k2 + 2][rest] == p0[rest]
        @test dp.pvec == p0
        empty!(ps.dynamic.load_step_events)
    end

    @testset "several steps, shed, restore, with a fault; solvers agree" begin
        for (t, sc) in ((0.05, 1.3), (0.15, 0.0), (0.3, 1.0))
            add_load_step_event!(ps, create_load_step_event(ps, bus_ext, ld.dtype.id, t, sc))
        end
        add_event!(ps, ContingencyEvent(4, 0.05, 0.1, 0.2))
        _, ref, _ = run(jacobian_reuse = false, predictor = false)
        for kw in ((;), (; solver = :schur), (; solver = :schur_gmres))
            _, tr, _ = run(; kw...)
            @test maximum(abs.(tr .- ref)) < 1e-8
            @test dp.pvec == p0
        end
        _, trc, _ = run(integrator = :crank_nicolson)        # different method: runs, restores
        @test all(isfinite, trc) && dp.pvec == p0
        # repeatable on the same ps
        _, ref2, _ = run(jacobian_reuse = false, predictor = false)
        @test ref2 == ref
        empty!(ps.dynamic.load_step_events); empty!(ps.dynamic.events)
        _, tr, _ = run()
        @test tr == tr0
    end

    @testset "restored after a failed run" begin
        add_load_step_event!(ps, create_load_step_event(ps, bus_ext, ld.dtype.id, dt, 1.5))
        add_event!(ps, ContingencyEvent(1, 0.0, 2dt, 0.1))           # rfault = 0: re-solve fails
        @test_throws Exception run(tf = 0.1)
        @test dp.pvec == p0
        empty!(ps.dynamic.load_step_events); empty!(ps.dynamic.events)
        foreach(GradPower.deactivate!, ps.dynamic.events)
    end

    @testset "event before the first step is rejected" begin
        add_load_step_event!(ps, create_load_step_event(ps, bus_ext, ld.dtype.id, 0.0, 1.2))
        @test_throws ArgumentError run(tf = 0.1)
        dp.zvec .= z0
        @test_throws ArgumentError GradPower.integrate_batched!(GradPower.BatchedLayout(dp, ps, 1), ps, 0.1)
        empty!(ps.dynamic.load_step_events)
    end

    @testset "integrate_batched! matches integrate!" begin
        add_load_step_event!(ps, create_load_step_event(ps, bus_ext, ld.dtype.id, 0.1, 1.25))
        add_event!(ps, ContingencyEvent(4, 0.05, 0.2, 0.25))
        zl = ps.dynamic.layout.zipload
        M = 3
        dp.zvec .= z0
        bl = GradPower.BatchedLayout(dp, ps, M)
        for m in 1:M, k in 1:zl.n, o in (0, 1, 7, 8)
            bl.p[m, Int(zl.par_ptr[k]) + o] *= 1 + 0.03 * (m - 1)
        end
        pb = copy(bl.p)
        tvb, trb = GradPower.integrate_batched!(bl, ps, 0.4)
        @test bl.p == pb
        for m in 1:M
            dp.zvec .= z0; dp.pvec .= bl.p[m, :]
            tv, tr = integrate!(dp, ps, 0.4)
            @test trb[m] == tr
            @test dp.pvec == bl.p[m, :]
        end
        dp.pvec .= p0
        empty!(ps.dynamic.load_step_events); empty!(ps.dynamic.events)
    end
end
