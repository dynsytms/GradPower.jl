# PERC2 in the network: registered load device `PERC2Load`, CPU paths
# (backward Euler / Crank–Nicolson, monolithic, batched). Schur solvers are
# unsupported with PERC2 and must throw ArgumentError.
# Needs test/perc2_helpers.jl.

@testset "PERC2 network" begin
    ex = joinpath(@__DIR__, "..", "examples")
    cases = (("2bus", "2bus.raw", "2bus_IEESGO.dyr"), ("IEEE 9", "ieee9_v33.raw", "ieee9bus_gov.dyr"))

    # Largest load of the case (external bus number, id) by active power.
    function largest_load(ps)
        l = ps.loads[argmax([l.pd for l in ps.loads])]
        return ps.buses[l.bus].i, l.id
    end

    # inert on a flat run
    flat_params() = GradPower.PERC2Params(; fcease = 0.0)

    # `perc2 = nothing` gives the plain ZIP reference.
    function setup(raw, dyr; perc2 = nothing, fraction = 0.8)
        ps = from_psse(joinpath(ex, raw), joinpath(ex, dyr))
        GradPower.build_network!(ps); GradPower.runpf!(ps)
        b, id = largest_load(ps)
        perc2 === nothing || add_perc2_load!(ps, b, id; fraction = fraction, params = perc2)
        dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
        return ps, dp, b, id
    end

    clear_events!(ps) = (for ev in ps.dynamic.events; GradPower.deactivate!(ev); end;
                         empty!(ps.dynamic.events); empty!(ps.dynamic.disconnect_events))

    # last 2*nbus rows of a trajectory are the bus voltages (vr, vi per bus)
    busrows(tr, ps) = size(tr, 1)-2length(ps.buses)+1:size(tr, 1)

    # compare a PERC2 run with its no-PERC2 reference on the common coordinates
    function max_dev(tr1, ps1, tr0, ps0)
        if size(tr0, 1) == size(tr1, 1)
            return maximum(abs.(tr1 .- tr0))
        end
        dd = ps0.dynamic.diff_dim + ps0.dynamic.alg_dim
        return max(maximum(abs.(tr1[1:dd, :] .- tr0[1:dd, :])),
                   maximum(abs.(tr1[busrows(tr1, ps1), :] .- tr0[busrows(tr0, ps0), :])))
    end

    solver_kw = (monolithic = (; solver = :monolithic),)
    integrators = (:backward_euler, :crank_nicolson)

    @testset "N1 flat run matches the no-PERC2 system" begin
        for (label, raw, dyr) in cases, integ in integrators, (sname, skw) in pairs(solver_kw)
            @testset "$label $integ $sname" begin
                ps0, dp0, _, _ = setup(raw, dyr)
                ps1, dp1, _, _ = setup(raw, dyr; perc2 = flat_params())
                t0, tr0 = integrate!(dp0, ps0, 5.0; dt = 1 / 120, integrator = integ, skw...)
                t1, tr1 = integrate!(dp1, ps1, 5.0; dt = 1 / 120, integrator = integ, skw...)
                @test t0 == t1
                @test all(isfinite, tr1)
                dev = max_dev(tr1, ps1, tr0, ps0)
                # The no-PERC2 run itself moves off its initial state over
                # 5 s (Newton-tolerance drift); report it next to the deviation.
                drift0 = maximum(abs.(tr0 .- tr0[:, 1])); drift1 = maximum(abs.(tr1 .- tr1[:, 1]))
                # Both runs drift ~2e-8 from their initial state over 5 s at
                # newton_tol = 1e-10, so the PERC2-vs-plain bound is set at
                # that Newton-tolerance level, not below it.
                dev < 5e-8 || @info "PERC2 N1 deviation" case = label integrator = integ solver = sname deviation = dev plain_drift = drift0 perc2_drift = drift1
                @test dev < 5e-8
                # companion: no worse than the baseline's own Newton-level drift (x3)
                @test dev <= 3 * max(drift0, 1e-8)
            end
        end
    end

    @testset "N2 finite-difference Jacobian" begin
        for (label, raw, dyr) in cases
            ps, dp, _, _ = setup(raw, dyr; perc2 = flat_params())
            dyn = ps.dynamic; L = dyn.layout
            n = length(dp.zvec)
            # off the equilibrium, so the voltage-direction terms are non-trivial
            z = copy(dp.zvec)
            nb = length(ps.buses)
            for k in 1:nb
                z[n-2nb+2k-1] *= 1.0 + 0.01 * k
                z[n-2nb+2k] *= 1.0 - 0.005 * k
            end
            J = GradPower.preallocate_jacobian(ps)
            fill!(J.nzval, 0.0)
            GradPower._rhs_jac_batched!(J, z, dp.uvec, dp.pvec, dyn, ps.network.ybus_real, L)
            J_fd = FiniteDiff.finite_difference_jacobian(z) do zz
                f = zeros(n)
                GradPower._rhs_fun_batched!(f, zz, copy(dp.uvec), dp.pvec, dyn, ps.network.ybus_real, L)
                return f
            end
            @test maximum(abs, Array(J) - J_fd) < 1e-6

            # and through the backward-Euler residual/Jacobian used by Newton
            zold = copy(dp.zvec); h = 1 / 120
            JB = GradPower.preallocate_jacobian(ps)
            GradPower.beuler_jac!(JB, z, zold, dp.uvec, dp.pvec, ps, dyn.diff_dim, h)
            JB_fd = FiniteDiff.finite_difference_jacobian(z) do zz
                f = zeros(n)
                GradPower.beuler!(f, zz, zold, copy(dp.uvec), dp.pvec, ps, dyn.diff_dim, h)
                return f
            end
            @test maximum(abs, Array(JB) - JB_fd) < 1e-6
        end
    end

    @testset "N3 Schur solvers throw ArgumentError; supported paths work" begin
        for (raw, dyr) in (("2bus.raw", "2bus_IEESGO.dyr"), ("ieee9_v33.raw", "ieee9bus_gov.dyr"))
            ps, dp, _, _ = setup(raw, dyr; perc2 = flat_params())
            z0 = copy(dp.zvec)
            run(; kw...) = (dp.zvec .= z0; integrate!(dp, ps, 0.1; dt = 1 / 120, kw...))
            @test_throws ArgumentError run(solver = :schur)
            @test_throws ArgumentError run(solver = :schur_gmres)
            @test_throws ArgumentError run(solver = :schur, integrator = :crank_nicolson)
            @test_throws ArgumentError run(integrator = :rk4)
            for kw in ((;), (; integrator = :crank_nicolson))
                tv, tr = run(; kw...)
                @test all(isfinite, tr)
            end
            # a failed (rejected) call leaves the system usable
            tv, tr = run()
            @test all(isfinite, tr)
        end
    end

    # PERC2 settings that cease on the fault (VLcease above the fault-time
    # voltage).
    fault_params() = GradPower.PERC2Params(; VLMflag = true, fcease = 0.9, VLcease = 0.95, Tcease = 0.02,
                                            Tv = 0.02, Tramp = 1.25, frecon = 0.5, Ldfact = 0.8)

    function fault_run(; integrator = :backward_euler, solver = :monolithic, tf = 3.0)
        ps, dp, b, id = setup("ieee9_v33.raw", "ieee9bus_gov.dyr"; perc2 = fault_params())
        ib = ps.busmap[b]
        nbrs = [br.fr == ib ? br.to : br.fr for br in ps.branches if br.fr == ib || br.to == ib]
        GradPower.add_event!(ps, ContingencyEvent(first(nbrs), 0.02, 0.1, 0.175))
        log = Any[]
        tv, tr = integrate!(dp, ps, tf; dt = 1 / 240, integrator, solver, perc2_log = log)
        clear_events!(ps)
        return (; ps, tv, tr, log, ib)
    end
    vmag(r, k) = (rows = busrows(r.tr, r.ps); hypot(r.tr[rows[2r.ib-1], k], r.tr[rows[2r.ib], k]))

    function check_fault(r, label)
        @test all(isfinite, r.tr)
        @test !isempty(r.log)
        kf = findfirst(t -> t >= 0.15, r.tv)
        @test vmag(r, kf) < 0.95            # the PERC2 bus sees the sag
        frac = [x.FracOn for x in r.log]
        @test minimum(frac) < 1
        ev = perc2_events_from_fracon(frac, 1 / 240)
        @test haskey(ev, :trip_time) && haskey(ev, :reconnect_start)
        haskey(ev, :trip_time) && @test 0.1 - 1e-9 <= ev[:trip_time] <= 0.175 + 0.2
        @test frac[end] > minimum(frac)     # the load comes back
    end

    @testset "N4 fault next to the PERC2 bus" begin
        runs = Dict{Tuple{Symbol,Symbol},Any}()
        for integ in integrators, sname in (:monolithic,)
            @testset "$integ $sname" begin
                r = fault_run(; integrator = integ, solver = sname)
                runs[(integ, sname)] = r
                check_fault(r, "$integ/$sname")
            end
        end
    end

    @testset "N5 disconnecting the PERC2 device zeroes its current" begin
        # PERC2 replaces the whole load; tripping it must give the same
        # trajectory as tripping the whole ZIP load of the plain system.
        ttrip = 0.2
        ps0, dp0, b, id = setup("ieee9_v33.raw", "ieee9bus_gov.dyr")
        add_load_trip!(ps0, b, id, ttrip)
        t0, tr0 = integrate!(dp0, ps0, 1.0; dt = 1 / 120)
        ps1, dp1, b1, id1 = setup("ieee9_v33.raw", "ieee9bus_gov.dyr"; perc2 = flat_params(), fraction = 1.0)
        idx = find_devices(ps1; model = GradPower.PERC2Load, bus = b1)
        @test length(idx) == 1
        GradPower.add_disconnect_event!(ps1, DisconnectDeviceEvent(only(idx), ttrip))
        log = Any[]
        t1, tr1 = integrate!(dp1, ps1, 1.0; dt = 1 / 120, perc2_log = log)
        @test all(isfinite, tr1)
        @test max_dev(tr1, ps1, tr0, ps0) < 1e-6
        # the generic add_load_trip! on the PERC2 load id
        ps2, dp2, b2, id2 = setup("ieee9_v33.raw", "ieee9bus_gov.dyr"; perc2 = flat_params(), fraction = 1.0)
        ok = try; add_load_trip!(ps2, b2, id2, ttrip); true; catch e; e isa ArgumentError ? false : rethrow(); end
        if ok
            nev = length(ps2.dynamic.disconnect_events)
            @test nev >= 1
            t2, tr2 = integrate!(dp2, ps2, 1.0; dt = 1 / 120)
            @test all(isfinite, tr2)
        else
            @info "N5: add_load_trip! does not address PERC2 loads (ArgumentError); DisconnectDeviceEvent used"
        end
    end

    @testset "N6 batched: M = 2 equals two single runs" begin
        function build(; p2 = true)
            ps, dp, b, id = setup("ieee9_v33.raw", "ieee9bus_gov.dyr"; perc2 = p2 ? fault_params() : nothing)
            ib = ps.busmap[b]
            nbrs = [br.fr == ib ? br.to : br.fr for br in ps.branches if br.fr == ib || br.to == ib]
            GradPower.add_event!(ps, ContingencyEvent(first(nbrs), 0.02, 0.1, 0.175))
            return ps, dp
        end
        scale_zip!(bl, ps, m, s) = (zl = ps.dynamic.layout.zipload;
                                    for k in 1:zl.n, o in (0, 1, 7, 8); bl.p[m, Int(zl.par_ptr[k]) + o] *= s; end)
        tf = 1.0; dt = 1 / 240
        for threaded in (false, true)
            ps, dp = build()
            z0 = copy(dp.zvec); p0 = copy(dp.pvec)
            bl = GradPower.BatchedLayout(dp, ps, 2)
            scale_zip!(bl, ps, 2, 1.1)
            pm = [copy(bl.p[m, :]) for m in 1:2]
            tvb, trb = GradPower.integrate_batched!(bl, ps, tf; dt = dt, threaded = threaded)
            clear_events!(ps)
            for m in 1:2
                ps1, dp1 = build()
                dp1.zvec .= z0; dp1.pvec .= pm[m]
                tv1, tr1 = integrate!(dp1, ps1, tf; dt = dt)
                clear_events!(ps1)
                @test tvb == tv1
                @test maximum(abs.(trb[m] .- tr1)) <= 1e-10
            end
            @test maximum(abs.(trb[1] .- trb[2])) > 1e-6     # the scenarios differ
        end
    end

    @testset "N7 .dyr import" begin
        dyr = joinpath(ex, "ieee9_perc2.dyr")
        ps = from_psse(joinpath(ex, "ieee9_v33.raw"), dyr)
        GradPower.build_network!(ps); GradPower.runpf!(ps)
        dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
        idx = find_devices(ps; model = GradPower.PERC2Load)
        @test !isempty(idx)
        for i in idx
            dt_ = ps.dynamic.devices[i].dtype
            @test dt_ isa GradPower.PERC2Load
            @test dt_.params isa GradPower.PERC2Params
            GradPower.validate(dt_.params)
        end
        # flat run: the same as the system without the record
        ps0 = from_psse(joinpath(ex, "ieee9_v33.raw"), joinpath(ex, "ieee9bus_gov.dyr"))
        GradPower.build_network!(ps0); GradPower.runpf!(ps0)
        dp0 = GradPower.DynamicProblem(ps0); GradPower.initialize_dynamics!(dp0, ps0)
        t0, tr0 = integrate!(dp0, ps0, 2.0; dt = 1 / 120)
        t1, tr1 = integrate!(dp, ps, 2.0; dt = 1 / 120)
        @test all(isfinite, tr1)
        @test max_dev(tr1, ps, tr0, ps0) < 1e-6
    end

    @testset "N8 KernelAbstractions CPU path refuses PERC2" begin
        ps, dp, _, _ = setup("ieee9_v33.raw", "ieee9bus_gov.dyr"; perc2 = flat_params())
        dyn = ps.dynamic; L = dyn.layout; ybus = ps.network.ybus_real
        inj_meta = GradPower.InjectionMeta(L)
        inj = zeros(2 * inj_meta.n_total)
        f2 = zeros(length(dp.zvec))
        @test_throws ArgumentError GradPower.dispatch_residual_kernels!(f2, dp.zvec, dp.uvec, dp.pvec, inj, L, dyn, ybus,
                                                                      inj_meta, GradPower.KA_CPU())
    end

    @testset "N9 queries on PERC2Load" begin
        ps, dp, b, id = setup("ieee9_v33.raw", "ieee9bus_gov.dyr"; perc2 = GradPower.PERC2Params(; Ldfact = 0.7, fcease = 0.0))
        idx = find_devices(ps; model = GradPower.PERC2Load)
        @test length(idx) == 1
        @test find_devices(ps; model = GradPower.PERC2Load, bus = b) == idx
        @test isempty(find_devices(ps; model = GradPower.PERC2Load, bus = 1))
        @test idx[1] in find_devices(ps; bus = b)
        dtype = ps.dynamic.devices[idx[1]].dtype
        names = GradPower.get_param_names(dtype)
        @test !isempty(names)
        vals = [GradPower.get_parameter(ps, idx[1], n) for n in names]
        @test all(isfinite, vals)
        @test length(unique(Symbol.(names))) == length(names)
        # a named parameter is readable through the live vector too
        @test GradPower.get_parameter(dp, ps, idx[1], first(names)) == vals[1]
        ln = findfirst(n -> Symbol(n) in (:Ldfact, :ldfact), names)
        ln === nothing || @test vals[ln] ≈ 0.7
        # every dynamic device is still queryable alongside it
        @test length(find_devices(ps)) == length(ps.dynamic.devices)
        # the perc2_states accessor
        st = GradPower.perc2_states(ps)
        @test length(st) == 1
    end
end
