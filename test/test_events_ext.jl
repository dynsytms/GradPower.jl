# Impedance faults (complex fault admittance) and removal events (line,
# generator with its controllers, load) addressed by PSS/E bus + id.

function _ev_setup(raw = "ieee9_v33.raw", dyr = "ieee9bus_gov.dyr")
    ex = joinpath(@__DIR__, "..", "examples")
    ps = from_psse(joinpath(ex, raw), joinpath(ex, dyr))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
    return ps, dp, copy(dp.zvec), copy(dp.pvec)
end

function _clear_all_events!(ps)
    d = ps.dynamic
    empty!(d.events); empty!(d.disconnect_events); empty!(d.trip_events); empty!(d.load_step_events)
end

@testset "Impedance faults" begin
    ps, dp, z0, p0 = _ev_setup()
    dyn = ps.dynamic; L = dyn.layout; ybus = ps.network.ybus_real
    n = length(z0); net = dyn.diff_dim + dyn.alg_dim
    rhs(z) = (f = zeros(n); GradPower._rhs_fun_batched!(f, z, copy(dp.uvec), dp.pvec, dyn, ybus, L); f)

    @testset "constructor" begin
        @test ContingencyEvent(4, 0.02, 0.1, 0.2).xfault == 0.0
        @test_throws ArgumentError ContingencyEvent(4, 0.0, 0.1, 0.2)               # bolted
        @test_throws ArgumentError ContingencyEvent(4, 0.0, 0.1, 0.2; xfault = 0.0)
        @test_throws ArgumentError ContingencyEvent(4, -0.01, 0.1, 0.2)
        @test_throws ArgumentError ContingencyEvent(4, Inf, 0.1, 0.2)
        @test ContingencyEvent(4, 0.0, 0.1, 0.2; xfault = 0.05).rfault == 0.0     # pure reactance
        g, b = GradPower.fault_admittance(ContingencyEvent(4, 0.01, 0.1, 0.2; xfault = 0.05))
        @test complex(g, b) ≈ 1 / complex(0.01, 0.05)
        @test GradPower.fault_admittance(ContingencyEvent(4, 0.02, 0.1, 0.2)) == (1 / 0.02, 0.0)
    end

    # perturbed state, so the fault current is generic
    z = z0 .+ 1e-3 .* sin.(1:n)
    for (r, x) in ((0.01, 0.05), (0.0, 0.03), (0.02, -0.01))
        ev = ContingencyEvent(4, r, 0.1, 0.2; xfault = x)
        @testset "residual = -y v at the bus, r=$r x=$x" begin
            f0 = rhs(z)
            push!(dyn.events, ev); GradPower.activate!(ev)
            f1 = rhs(z)
            y = 1 / complex(r, x)
            v = complex(z[net + 7], z[net + 8])                 # bus 4
            I = y * v
            d = f1 .- f0
            @test d[net + 7] ≈ -real(I) rtol = 1e-12
            @test d[net + 8] ≈ -imag(I) rtol = 1e-12
            d[net + 7] = 0.0; d[net + 8] = 0.0
            @test all(iszero, d)
            # Jacobian = finite differences (with the fault on)
            J = GradPower.preallocate_jacobian(ps)
            GradPower.beuler_jac_batched!(J, z, copy(dp.uvec), dp.pvec, dyn, ybus, L, dyn.diff_dim, 0.0)
            resid(x) = (r = zeros(n); GradPower.beuler_batched!(r, x, z, copy(dp.uvec), dp.pvec, dyn,
                                                                 ybus, L, dyn.diff_dim, 0.0); r)
            Jfd = FiniteDiff.finite_difference_jacobian(resid, z)
            @test maximum(abs.(Matrix(J) .- Jfd)) < 1e-5
            GradPower.deactivate!(ev); empty!(dyn.events)
        end
    end

    @testset "trajectories" begin
        run(; kw...) = (dp.zvec .= z0; integrate!(dp, ps, 0.4; kw...)[2])
        add_event!(ps, ContingencyEvent(4, 0.02, 0.1, 0.2))
        tr_r = run()
        empty!(dyn.events)
        add_event!(ps, ContingencyEvent(4, 0.02, 0.1, 0.2; xfault = 0.0))
        @test run() == tr_r                                    # xfault = 0 == old fault, bitwise
        empty!(dyn.events)
        add_event!(ps, ContingencyEvent(4, 0.01, 0.1, 0.2; xfault = 0.05))
        ref = run(jacobian_reuse = false, predictor = false)
        @test maximum(abs.(ref .- tr_r)) > 1e-3
        for kw in ((;), (; solver = :schur), (; solver = :schur_gmres))
            @test maximum(abs.(run(; kw...) .- ref)) < 1e-8
        end
        # an inductive fault depresses the faulted bus voltage during the fault
        k = Int(round(0.15 * 120)) + 1
        vm(t) = hypot(t[net + 7], t[net + 8])
        @test vm(ref[:, k]) < 0.9 * vm(ref[:, 1])
        empty!(dyn.events)
    end
end

@testset "Removal events" begin
    @testset "find_branch / line removal by circuit id" begin
        ps, dp, z0, _ = _ev_setup("IEEE39.raw", "IEEE39_gov.dyr")
        # Build a parallel circuit: duplicate an existing line as circuit "2".
        b1 = ps.branches[1]
        fe, te = ps.buses[b1.fr].i, ps.buses[b1.to].i
        @test find_branch(ps, fe, te) == 1
        @test find_branch(ps, te, fe) == 1                     # either direction
        @test find_branch(ps, fe, te; ckt = "1") == 1
        @test find_branch(ps, fe, te; ckt = 1) == 1
        @test_throws ArgumentError find_branch(ps, fe, te; ckt = "9")
        @test_throws ArgumentError find_branch(ps, fe, 99999)
        push!(ps.branches, GradPower.Branch(b1.fr, b1.to, "'2 '", b1.r, b1.x, b1.sh, b1.tap, b1.shift))
        err = try find_branch(ps, fe, te); nothing catch e; e end
        @test err isa ArgumentError && occursin("pass ckt", err.msg)
        @test find_branch(ps, fe, te; ckt = "2") == length(ps.branches)
        @test find_branch(ps, fe, te; ckt = "'1'") == 1
        tev = create_trip_line_event(ps, fe, te, 0.1; ckt = "2")
        @test tev.branch == length(ps.branches)
        @test_throws ArgumentError create_trip_line_event(ps, fe, te, 0.1)
        add_trip_event!(ps, create_trip_line_event(ps, fe, te, 0.1; ckt = "1"))
        @test_throws ArgumentError add_trip_event!(ps, create_trip_line_event(ps, fe, te, 0.2; ckt = "1"))
        empty!(ps.dynamic.trip_events)
        pop!(ps.branches)
    end

    @testset "line removal = network without the line" begin
        # After the trip, the post-trip equilibrium equals the one of the
        # case without that line in the network (Ybus rebuilt).
        ps, dp, z0, _ = _ev_setup("IEEE39.raw", "IEEE39_gov.dyr")
        b = ps.branches[findfirst(b -> b.tap == 0.0, ps.branches)]
        fe, te = ps.buses[b.fr].i, ps.buses[b.to].i
        y0 = copy(ps.network.ybus_real)
        add_trip_event!(ps, create_trip_line_event(ps, fe, te, 0.1))
        GradPower._apply_trip_line!(ps.network.ybus_real, ps.dynamic.trip_events[1])
        yt = copy(ps.network.ybus_real)
        copyto!(ps.network.ybus_real, y0)
        # rebuild without the branch
        bi = find_branch(ps, fe, te)
        br = popat!(ps.branches, bi)
        GradPower.build_network!(ps)
        @test maximum(abs.(Matrix(ps.network.ybus_real) .- Matrix(yt))) < 1e-10
        insert!(ps.branches, bi, br); GradPower.build_network!(ps)
        @test ps.network.ybus_real == y0
        dp.zvec .= z0
        tv, tr = integrate!(dp, ps, 0.3)
        @test all(isfinite, tr)
        @test ps.network.ybus_real == y0                        # restored
        @test maximum(abs.(tr[:, end] .- z0)) > 1e-4             # the trip acts
        empty!(ps.dynamic.trip_events)
    end

    @testset "generator removal takes its controllers offline" begin
        ps, dp, z0, _ = _ev_setup()
        devs = ps.dynamic.devices; dd = ps.dynamic.diff_dim
        idx = generator_devices(ps, 2, "1")
        types = Set(typeof(devs[i].dtype) for i in idx)
        @test types == Set([GradPower.Genrou, GradPower.IEESGO])
        @test all(devs[i].dtype.bus == 2 for i in idx)
        @test_throws ArgumentError generator_devices(ps, 2, "7")
        @test_throws ArgumentError generator_devices(ps, 99999, "1")
        online0 = [d.online for d in devs]; y0 = copy(ps.network.ybus_real)
        evs = add_generator_trip!(ps, 2, "'1'", 0.1)
        @test sort([e.device_idx for e in evs]) == sort(idx)
        kt = Int(round(0.1 * 120)) + 1                           # column right after the trip
        rows = reduce(vcat, [vcat(devs[i].diff_ptr:devs[i].diff_ptr + devs[i].dtype.diff_size - 1,
                                  dd + devs[i].alg_ptr:dd + devs[i].alg_ptr + devs[i].dtype.alg_size - 1)
                             for i in idx])
        run(; kw...) = (dp.zvec .= z0; integrate!(dp, ps, 0.5; kw...)[2])
        ref = run(jacobian_reuse = false, predictor = false)
        @test all(isfinite, ref)
        @test all(ref[rows, end] .== ref[rows, kt + 1])          # frozen from the trip on
        @test maximum(abs.(ref[:, end] .- z0)) > 1e-3            # the system reacts
        for kw in ((;), (; solver = :schur), (; solver = :schur_gmres))
            @test maximum(abs.(run(; kw...) .- ref)) < 1e-8
        end
        @test [d.online for d in devs] == online0                # restored
        @test all(all(t.online) for t in values(ps.dynamic.layout.tables))
        empty!(ps.dynamic.disconnect_events)
        @test run() == (dp.zvec .= z0; integrate!(dp, ps, 0.5)[2])
    end

    @testset "load removal == load step to zero" begin
        ps, dp, z0, p0 = _ev_setup()
        li = findfirst(d -> d.dtype isa GradPower.ZIPLoad, ps.dynamic.devices)
        ld = ps.dynamic.devices[li]; be = ps.buses[ld.dtype.bus].i
        ev = add_load_trip!(ps, be, ld.dtype.id, 0.1)
        @test ev.device_idx == li
        @test_throws ArgumentError add_load_trip!(ps, be, "ZZ", 0.1)
        dp.zvec .= z0; trA = integrate!(dp, ps, 0.4)[2]
        empty!(ps.dynamic.disconnect_events)
        add_load_step_event!(ps, create_load_step_event(ps, be, ld.dtype.id, 0.1, 0.0))
        dp.zvec .= z0; trB = integrate!(dp, ps, 0.4)[2]
        empty!(ps.dynamic.load_step_events)
        @test maximum(abs.(trA .- trB)) < 1e-10
        @test maximum(abs.(trA[:, end] .- z0)) > 1e-3
        @test all(d.online for d in ps.dynamic.devices) && dp.pvec == p0
    end

    @testset "event before the first step is rejected" begin
        ps, dp, z0, _ = _ev_setup()
        add_generator_trip!(ps, 2, "1", 0.0)
        @test_throws ArgumentError (dp.zvec .= z0; integrate!(dp, ps, 0.1))
        _clear_all_events!(ps)
        add_trip_event!(ps, create_trip_line_event(ps, 4, 5, 0.0))
        @test_throws ArgumentError (dp.zvec .= z0; integrate!(dp, ps, 0.1))
        _clear_all_events!(ps)
    end

    @testset "integrate_batched! with removals == integrate!" begin
        ps, dp, z0, p0 = _ev_setup()
        add_generator_trip!(ps, 3, "1", 0.1)
        add_trip_event!(ps, create_trip_line_event(ps, 4, 5, 0.15; toff = 0.3))
        add_load_trip!(ps, 5, ps.dynamic.devices[findfirst(d -> d.dtype isa GradPower.ZIPLoad &&
                                         d.dtype.bus == ps.busmap[5], ps.dynamic.devices)].dtype.id, 0.2)
        add_event!(ps, ContingencyEvent(7, 0.01, 0.05, 0.1; xfault = 0.02))
        zl = ps.dynamic.layout.zipload
        M = 3
        dp.zvec .= z0
        bl = GradPower.BatchedLayout(dp, ps, M)
        for m in 1:M, k in 1:zl.n, o in (0, 1, 7, 8)
            bl.p[m, Int(zl.par_ptr[k]) + o] *= 1 + 0.03 * (m - 1)
        end
        y0 = copy(ps.network.ybus_real)
        for threaded in (false, true)
            tvb, trb = GradPower.integrate_batched!(bl, ps, 0.4; threaded = threaded)
            @test ps.network.ybus_real == y0
            @test all(d.online for d in ps.dynamic.devices)
            @test !any(ev.status for ev in ps.dynamic.events)
            for m in 1:M
                dp.zvec .= z0; dp.pvec .= bl.p[m, :]
                @test trb[m] == integrate!(dp, ps, 0.4)[2]
            end
            bl.z .= z0'
        end
        dp.pvec .= p0
        _clear_all_events!(ps)
    end

    @testset "static generator removal (ACTIVSg2000)" begin
        ps, dp, z0, _ = _ev_setup("ACTIVSg2000.raw", "ACTIVSg2000.dyr")
        devs = ps.dynamic.devices
        idx = generator_devices(ps, 6085, "1")
        @test length(idx) == 1 && devs[idx[1]].dtype isa GradPower.StaticGenerator
        @test_throws ArgumentError generator_devices(ps, 1090, "1")   # aggregated with another gen
        # a machine with exciter + stabilizer
        idx2 = generator_devices(ps, 3106, "1")
        @test Set(typeof(devs[i].dtype) for i in idx2) ⊇ Set([GradPower.Genrou, GradPower.ESDC1A, GradPower.IEEEST])
        add_generator_trip!(ps, 6085, "1", 0.05)
        add_generator_trip!(ps, 3106, "1", 0.1)
        dp.zvec .= z0
        tv, tr = integrate!(dp, ps, 0.3)
        @test all(isfinite, tr)
        @test all(d.online for d in devs)
        _clear_all_events!(ps)
    end
end
