# Selective trajectory recording (save_every / save_idxs / save_eltype /
# step_callback) for integrate! and integrate_batched!. Every reduced output
# must equal the corresponding slice of the full trajectory of an identical
# run, and the options must not change the computed solution.

function _rec_setup(raw = "ieee9_v33.raw", dyr = "ieee9bus_gov.dyr")
    ex = joinpath(@__DIR__, "..", "examples")
    ps = from_psse(joinpath(ex, raw), joinpath(ex, dyr))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    return ps, dp, copy(dp.zvec)
end

_full(dp, ps, z0, tf; kw...) = (dp.zvec .= z0; r = integrate!(dp, ps, tf; kw...); (r..., copy(dp.zvec)))

@testset "integrate! recorder" begin
    ps, dp, z0 = _rec_setup()
    add_event!(ps, ContingencyEvent(4, 0.02, 0.1, 0.2))
    n = length(z0)
    tf = 0.5                                   # 60 steps, events at steps 12 and 24
    tv, tr, zf = _full(dp, ps, z0, tf)
    @test size(tr) == (n, 61) && length(tv) == 61

    @testset "defaults reproduce the full trajectory" begin
        tv2, tr2, zf2 = _full(dp, ps, z0, tf; save_every = 1, save_idxs = :, save_eltype = Float64)
        @test tv2 == tv && tr2 == tr && zf2 == zf
    end

    @testset "save_every = $s (divisible and not)" for s in (1, 2, 5, 7, 60, 61, 100)
        tvs, trs, zfs = _full(dp, ps, z0, tf; save_every = s)
        cols = 1:s:61
        @test tvs == tv[cols]
        @test trs == tr[:, cols]
        @test zfs == zf                          # final state independent of storage
        @test tvs[1] == 0.0 && trs[:, 1] == z0   # initial state always stored
    end

    @testset "event samples follow the full-trajectory convention" begin
        # stride 12 lands on the event steps (t = 0.1, 0.2)
        tvs, trs, _ = _full(dp, ps, z0, tf; save_every = 12)
        @test tvs ≈ [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        @test trs == tr[:, 1:12:61]
    end

    @testset "save_idxs" begin
        net = ps.dynamic.diff_dim + ps.dynamic.alg_dim
        for idx in ([3], [n, 1, 5, 5], collect(net+1:n), 1:2:n)
            tvs, trs, zfs = _full(dp, ps, z0, tf; save_idxs = idx, save_every = 3)
            @test trs == tr[collect(idx), 1:3:61]
            @test zfs == zf
        end
        @test_throws ArgumentError integrate!((dp.zvec .= z0; dp), ps, tf; save_idxs = [0])
        @test_throws ArgumentError integrate!((dp.zvec .= z0; dp), ps, tf; save_idxs = [n + 1])
        @test_throws ArgumentError integrate!((dp.zvec .= z0; dp), ps, tf; save_every = 0)
    end

    @testset "save_eltype = Float32" begin
        tvs, trs, zfs = _full(dp, ps, z0, tf; save_eltype = Float32, save_every = 4)
        @test eltype(trs) == Float32
        @test trs == Float32.(tr[:, 1:4:61])
        @test zfs == zf                          # state itself stays Float64
    end

    @testset "step_callback sees every step, at full resolution" begin
        ks = Int[]; ts = Float64[]; zs = Vector{Float64}[]
        cb = (k, t, z) -> (push!(ks, k); push!(ts, t); push!(zs, copy(z)))
        tvs, trs, zfs = _full(dp, ps, z0, tf; save_every = 7, step_callback = cb)
        @test ks == 1:61
        @test ts == tv
        @test reduce(hcat, zs) == tr
        @test trs == tr[:, 1:7:61] && zfs == zf
    end

    @testset "solver options" begin
        for kw in ((; solver = :schur), (; jacobian_reuse = false, predictor = false))
            tvf, trf, zff = _full(dp, ps, z0, tf; kw...)
            tvs, trs, zfs = _full(dp, ps, z0, tf; save_every = 4, save_idxs = 2:9, kw...)
            @test trs == trf[2:9, 1:4:61] && zfs == zff
        end
    end

    @testset "zero-length and one-step horizons" begin
        tv0, tr0, zf0 = _full(dp, ps, z0, 0.0; save_every = 3, save_idxs = [1, 2])
        @test tv0 == [0.0] && tr0 == reshape(z0[[1, 2]], 2, 1) && zf0 == z0
        tvf, trf, zff = _full(dp, ps, z0, 1/120)
        tv1, tr1, zf1 = _full(dp, ps, z0, 1/120; save_every = 5)
        @test tv1 == [0.0] && tr1 == trf[:, 1:1] && zf1 == zff
    end
    empty!(ps.dynamic.events)

    @testset "no events" begin
        tvf, trf, zff = _full(dp, ps, z0, 0.3)
        tvs, trs, zfs = _full(dp, ps, z0, 0.3; save_every = 4, save_eltype = Float32)
        @test trs == Float32.(trf[:, 1:4:end]) && zfs == zff
    end

    @testset "event on the last step" begin
        add_event!(ps, ContingencyEvent(4, 0.02, 0.1, 0.25))   # toff = tf
        tvf, trf, zff = _full(dp, ps, z0, 0.25)
        tvs, trs, zfs = _full(dp, ps, z0, 0.25; save_every = 2)
        @test trs == trf[:, 1:2:end] && zfs == zff
        @test zff == trf[:, end]                 # final state = last stored column convention
        empty!(ps.dynamic.events)
    end
end

@testset "integrate! recorder with TGOV1 limits" begin
    # record_limit_events! needs the previous accepted state, which no longer
    # comes from the stored trajectory: storing less must not change limit
    # modes, events or the solution.
    ex = joinpath(@__DIR__, "..", "examples")
    ps = from_psse(joinpath(ex, "2bus.raw"), joinpath(ex, "2bus_TGOV1.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        d.dtype isa GradPower.ZIPLoad && (d.dtype.α = 0.5)
    end
    dp = DynamicProblem(ps); initialize_dynamics!(dp, ps)
    t = ps.dynamic.layout.tgov1
    pp = Int(t.par_ptr[1]); st = Int(t.diff_ptr[1]) + 1
    dp.pvec[pp + 2] = dp.zvec[st] + 0.01
    dp.pvec[pp + 3] = dp.zvec[st] - 0.01
    dp.pvec[pp + 7] += 0.05
    z0 = copy(dp.zvec)
    for method in (:active_set, :complementarity)
        ev_full = LimitEvent[]; ev_red = LimitEvent[]
        tvf, trf, zff = _full(dp, ps, z0, 0.05; dt = 0.005, limit_method = method, limit_events = ev_full)
        tvs, trs, zfs = _full(dp, ps, z0, 0.05; dt = 0.005, limit_method = method, limit_events = ev_red,
                              save_every = 3, save_idxs = [st])
        @test !isempty(ev_full)
        @test ev_red == ev_full
        @test trs == trf[[st], 1:3:end] && zfs == zff
    end
end

@testset "integrate_batched! recorder" begin
    ps, dp, z0 = _rec_setup()
    add_event!(ps, ContingencyEvent(4, 0.02, 0.1, 0.2))
    zl = ps.dynamic.layout.zipload
    M = 4
    function bl()
        dp.zvec .= z0
        b = GradPower.BatchedLayout(dp, ps, M)
        for m in 1:M, k in 1:zl.n, o in (0, 1, 7, 8)   # pl, ql, yreal, yimag: scenarios differ
            b.p[m, Int(zl.par_ptr[k]) + o] *= 1 + 0.03 * (m - 1)
        end
        b
    end
    b0 = bl(); tvf, trf = GradPower.integrate_batched!(b0, ps, 0.5; threaded = false)
    @test maximum(abs.(trf[M] .- trf[1])) > 1e-4
    idx = [1, 4, length(z0)]
    for threaded in (false, true)
        b = bl()
        hits = [Int[] for _ in 1:M]
        cb = (m, k, t, z) -> (push!(hits[m], k); z == trf[m][:, k] || error("callback state mismatch m=$m k=$k"))
        tvs, trs = GradPower.integrate_batched!(b, ps, 0.5; threaded = threaded, save_every = 7,
                                                 save_idxs = idx, save_eltype = Float32, step_callback = cb)
        @test tvs == tvf[1:7:end]
        for m in 1:M
            @test trs[m] == Float32.(trf[m][idx, 1:7:end])
            @test hits[m] == 1:61
        end
        @test b.z == b0.z
    end
    empty!(ps.dynamic.events)
end
