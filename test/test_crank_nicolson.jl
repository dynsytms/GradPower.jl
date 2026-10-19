# Crank–Nicolson (trapezoidal) integrator: residual/Jacobian definition,
# equilibrium, convergence order, events, solver/batched consistency.

function _cn_setup(raw = "ieee9_v33.raw", dyr = "ieee9bus_gov.dyr")
    ex = joinpath(@__DIR__, "..", "examples")
    ps = from_psse(joinpath(ex, raw), joinpath(ex, dyr))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    return ps, dp, copy(dp.zvec), copy(dp.pvec)
end

_cn_run(dp, ps, z0, tf; kw...) = (dp.zvec .= z0; integrate!(dp, ps, tf; kw...))

# Governor reference step: a smooth transient (no discontinuity after t = 0).
function _pref_step!(dp, ps, d)
    ig = ps.dynamic.layout.ieesgo
    for k in 1:ig.n
        dp.pvec[Int(ig.par_ptr[k]) + 11] += d
    end
end

@testset "Crank–Nicolson" begin
    ps, dp, z0, p0 = _cn_setup()
    dyn = ps.dynamic; L = dyn.layout; ybus = ps.network.ybus_real
    n = length(z0); h = 1/120
    isdiff = falses(n)
    isdiff[dyn.diff_indices === nothing ? (1:dyn.diff_dim) : dyn.diff_indices] .= true
    rhs(z) = (f = zeros(n); u = copy(dp.uvec); GradPower._rhs_fun_batched!(f, z, u, dp.pvec, dyn, ybus, L); f)

    @testset "argument checks" begin
        @test_throws ArgumentError _cn_run(dp, ps, z0, 0.1; integrator = :rk4)
        @test_throws ArgumentError _cn_run(dp, ps, z0, 0.1; integrator = :crank_nicolson,
                                           limit_method = :active_set)
    end

    @testset "default is backward Euler, unchanged" begin
        tv1, tr1 = _cn_run(dp, ps, z0, 0.2)
        tv2, tr2 = _cn_run(dp, ps, z0, 0.2; integrator = :backward_euler)
        @test tv1 == tv2 && tr1 == tr2
    end

    # A state off the equilibrium: 0.15 s into a pref-step transient.
    _pref_step!(dp, ps, 0.05)
    _, trs = _cn_run(dp, ps, z0, 0.15)
    zk = trs[:, end - 1]; z = trs[:, end] .+ 1e-4 .* sin.(1:n)

    @testset "residual = z - z_k - h/2 (f(z) + f(z_k)), g(z) on alg rows" begin
        a = similar(z); fk = similar(z); u = copy(dp.uvec)
        GradPower._cn_anchor!(a, fk, zk, u, dp.pvec, dyn, ybus, L, h)
        r = zeros(n)
        GradPower.beuler_batched!(r, z, a, copy(dp.uvec), dp.pvec, dyn, ybus, L, dyn.diff_dim, h / 2)
        fz = rhs(z); fzk = rhs(zk)
        expect = [isdiff[i] ? z[i] - zk[i] - h / 2 * (fz[i] + fzk[i]) : fz[i] for i in 1:n]
        @test maximum(abs.(r .- expect)) < 1e-12
    end

    @testset "Jacobian = finite differences (I - h/2 f_z on diff rows)" begin
        for zz in (z0, z)
            a = similar(zz); fk = similar(zz)
            GradPower._cn_anchor!(a, fk, zk, copy(dp.uvec), dp.pvec, dyn, ybus, L, h)
            resid(x) = (r = zeros(n); GradPower.beuler_batched!(r, x, a, copy(dp.uvec), dp.pvec,
                                                                 dyn, ybus, L, dyn.diff_dim, h / 2); r)
            J = GradPower.preallocate_jacobian(ps)
            GradPower.beuler_jac_batched!(J, zz, copy(dp.uvec), dp.pvec, dyn, ybus, L, dyn.diff_dim, h / 2)
            Jfd = FiniteDiff.finite_difference_jacobian(resid, zz)
            @test maximum(abs.(Matrix(J) .- Jfd)) < 1e-5
        end
    end
    dp.pvec .= p0

    @testset "flat line ($case)" for (case, raw, dyr) in (("2bus", "2bus.raw", "2bus.dyr"),
                                                         ("ieee9", "ieee9_v33.raw", "ieee9bus_gov.dyr"))
        ps2, dp2, z02, _ = _cn_setup(raw, dyr)
        _, tr = _cn_run(dp2, ps2, z02, 2.0; integrator = :crank_nicolson)
        @test maximum(abs.(tr .- z02)) < 1e-8
    end

    @testset "order: CN 2, BE 1 (smooth transient)" begin
        _pref_step!(dp, ps, 0.05)
        tf = 0.5
        zref = _cn_run(dp, ps, z0, tf; dt = 1/1920, integrator = :crank_nicolson)[2][:, end]
        err(m, hh) = maximum(abs.(_cn_run(dp, ps, z0, tf; dt = hh, integrator = m)[2][:, end] .- zref))
        hs = (1/40, 1/80, 1/160)
        ecn = [err(:crank_nicolson, hh) for hh in hs]
        ebe = [err(:backward_euler, hh) for hh in hs]
        @test all(3.5 .< ecn[1:2] ./ ecn[2:3] .< 4.6)
        @test all(1.7 .< ebe[1:2] ./ ebe[2:3] .< 2.3)
        @test ecn[end] < ebe[end] / 10
        dp.pvec .= p0
    end

    add_event!(ps, ContingencyEvent(4, 0.02, 0.1, 0.2))
    @testset "fault: algebraic consistency and convergence" begin
        tv, tr = _cn_run(dp, ps, z0, 0.6; integrator = :crank_nicolson)
        @test all(isfinite, tr)
        # Each stored state solves g = 0 with the equations of its interval
        # (stored states at event steps are pre-event, as for BE).
        alg = findall(!, isdiff)
        for (j, t) in enumerate(tv)
            j == 1 && continue
            for ev in ps.dynamic.events
                ev.ton < t - 1e-9 && t <= ev.toff + 1e-9 ? GradPower.activate!(ev) : GradPower.deactivate!(ev)
            end
            @test maximum(abs.(rhs(tr[:, j])[alg])) < 1e-9
        end
        foreach(GradPower.deactivate!, ps.dynamic.events)
        # converges to a fine CN run on the common grid
        _, trf = _cn_run(dp, ps, z0, 0.6; dt = 1/960, integrator = :crank_nicolson)
        e120 = maximum(abs.(tr .- trf[:, 1:8:end]))
        _, tr2 = _cn_run(dp, ps, z0, 0.6; dt = 1/240, integrator = :crank_nicolson)
        e240 = maximum(abs.(tr2 .- trf[:, 1:4:end]))
        @test e240 < e120 / 1.5
    end

    @testset "solvers / Newton options agree" begin
        _, ref = _cn_run(dp, ps, z0, 0.4; integrator = :crank_nicolson,
                         jacobian_reuse = false, predictor = false)
        for kw in ((;), (; solver = :schur), (; solver = :schur_gmres),
                   (; solver = :schur, jacobian_reuse = false, predictor = false))
            _, tr = _cn_run(dp, ps, z0, 0.4; integrator = :crank_nicolson, kw...)
            @test maximum(abs.(tr .- ref)) < 1e-8
        end
    end

    @testset "integrate_batched! matches integrate!" begin
        zl = ps.dynamic.layout.zipload
        M = 3
        dp.zvec .= z0
        bl = GradPower.BatchedLayout(dp, ps, M)
        for m in 1:M, k in 1:zl.n, o in (0, 1, 7, 8)
            bl.p[m, Int(zl.par_ptr[k]) + o] *= 1 + 0.03 * (m - 1)
        end
        tvb, trb = GradPower.integrate_batched!(bl, ps, 0.4; integrator = :crank_nicolson)
        @test trb[1] != trb[M]
        for m in 1:M
            dp.zvec .= z0; dp.pvec .= bl.p[m, :]
            tv, tr = integrate!(dp, ps, 0.4; integrator = :crank_nicolson)
            @test tvb == tv && trb[m] == tr && bl.z[m, :] == tr[:, end]
        end
        dp.pvec .= p0
        @test_throws ArgumentError GradPower.integrate_batched!(GradPower.BatchedLayout(dp, ps, 1),
                                                                ps, 0.1; integrator = :foo)
    end
    empty!(ps.dynamic.events)
end
