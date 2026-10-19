# Early termination (`stop_condition`) in integrate!: the output must be the
# exact prefix of the full run, the final state the stop-step state, and the
# system must be restored (events off) so later runs are unaffected.

@testset "integrate! stop_condition" begin
    ex = joinpath(@__DIR__, "..", "examples")
    ps = from_psse(joinpath(ex, "ieee9_v33.raw"), joinpath(ex, "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
    z0 = copy(dp.zvec)
    add_event!(ps, ContingencyEvent(4, 0.02, 0.1, 0.3))
    run(; kw...) = (dp.zvec .= z0; r = integrate!(dp, ps, 0.5; kw...); (r..., copy(dp.zvec)))
    tv, tr, zf = run()

    @testset "never true == no condition" begin
        calls = Int[]
        tv2, tr2, zf2 = run(stop_condition = (k, t, z) -> (push!(calls, k); false))
        @test tv2 == tv && tr2 == tr && zf2 == zf
        @test calls == 2:61
    end

    @testset "stop at step $K, save_every $s" for K in (2, 13, 20, 37, 61), s in (1, 4)
        seen = Ref(0)
        tvs, trs, zfs = run(save_every = s, stop_condition = (k, t, z) -> (seen[] = k; k == K))
        @test seen[] == K                        # not called after the stop
        @test tvs == tv[1:s:K]
        @test trs == tr[:, 1:s:K]
        @test zfs == tr[:, K]                    # dp.zvec = stop-step state
        @test tvs[end] <= tv[K]
    end

    @testset "state-based condition, CN, Schur" begin
        w = GradPower.gen_speeds(ps)[1]
        thr = 0.5 * maximum(abs.(tr[w, :]))
        Kref = findfirst(j -> abs(tr[w, j]) > thr, 2:61) + 1
        cond = (k, t, z) -> abs(z[w]) > thr
        tvs, trs, zfs = run(stop_condition = cond)
        @test size(trs, 2) == Kref && trs == tr[:, 1:Kref]
        for kw in ((; integrator = :crank_nicolson), (; solver = :schur))
            tvf, trf, _ = run(; kw...)
            K = findfirst(j -> abs(trf[w, j]) > thr, 2:61) + 1
            tvs, trs, zfs = run(; stop_condition = cond, kw...)
            @test trs == trf[:, 1:K] && zfs == trf[:, K]
        end
    end

    @testset "stop during a fault restores the system" begin
        tvs, trs, _ = run(stop_condition = (k, t, z) -> t >= 0.2 - 1e-12)   # fault active
        @test tvs[end] ≈ 0.2
        @test !any(ev -> ev.status, ps.dynamic.events)
        tv3, tr3, zf3 = run()                    # same ps, full run: unchanged
        @test tv3 == tv && tr3 == tr && zf3 == zf
    end
    empty!(ps.dynamic.events)
end
