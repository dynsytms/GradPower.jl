# PERC2 core: reference playback (test/perc2_reference/) and spec-derived
# tests. Needs test/perc2_helpers.jl.

@testset "PERC2 core" begin
    names = perc2_case_names()

    ref_dt_tol = 1e-9

    @testset "reference harness" begin
        ncases = length(names)
        @test ncases >= 20
        for n in names
            c = load_case(n)
            @test c.ref.t[1] == 0.0
            @test length(c.ref.t) == round(Int, c.t_end / c.dt) + 1
            @test maximum(abs, c.ref.t .- (0:length(c.ref.t)-1) .* c.dt) < 1e-9
            # profile convention reproduces the reference V column
            vprof = [perc2_profile(c.breakpoints, t)[1] for t in c.ref.t]
            @test maximum(abs, vprof .- c.ref.V) < 1e-12
            # the event extraction used below agrees with the reference events
            ev = perc2_events_from_fracon(c.ref.FracOn, c.dt)
            @test sort!(collect(keys(ev))) == sort!(collect(keys(c.events)))
            for k in keys(c.events)
                tol = k === :reconnect_end ? c.dt + 1e-9 : 1e-9
                @test abs(ev[k] - c.events[k]) <= tol
            end
        end
        @test isa(params_from_cons(load_case(first(names)).cons), GradPower.PERC2Params)
    end

    @testset "reference playback" begin
        for n in names
            c = load_case(n)
            @testset "$n" begin
                p = params_from_cons(c.cons)
                snaps = playback(p, c)
                @test length(snaps) == length(c.ref.t)
                vtf = [s.Vtf for s in snaps]
                ppu = [s.Ppu0 for s in snaps]
                frac = [s.FracOn for s in snaps]
                for (lab, got, ref) in (("Vtf", vtf, c.ref.Vtf), ("P", ppu, c.ref.P),
                                        ("FracOn", frac, c.ref.FracOn))
                    err = abs.(got .- ref)
                    mx = maximum(err)
                    if !(mx < ref_dt_tol)
                        k = findfirst(>=(ref_dt_tol), err)
                        @info "PERC2 reference mismatch" case = n quantity = lab step = k - 1 t = c.ref.t[k] got = got[k] expected = ref[k] maxerr = mx
                    end
                    @test mx < ref_dt_tol
                end
                # events from the FracOn trace, compared with the reference's
                ev = perc2_events_from_fracon(frac, c.dt)
                @test sort!(collect(keys(ev))) == sort!(collect(keys(c.events)))
                for k in keys(c.events)
                    haskey(ev, k) || continue
                    # reconnect_end: the reference logs it from its float ramp
                    # timer, which can fire one step after FracOn has reached its
                    # final value (s2_vlm_ldfact03); allow that one step.
                    tol = k === :reconnect_end ? c.dt + 1e-9 : 1e-9
                    ok = abs(ev[k] - c.events[k]) <= tol
                    ok || @info "PERC2 event mismatch" case = n event = k got = ev[k] expected = c.events[k]
                    @test ok
                end
                # same-step check against the event derived from the reference trace
                evref = perc2_events_from_fracon(c.ref.FracOn, c.dt)
                @test ev == evref
            end
        end
    end

    # --- spec-derived tests (independent of the reference) -----------------
    s1_cons = load_case("s1_nocease_sag06").cons
    s2_cons = load_case("s2_vlm_dip_short").cons

    @testset "T1 flat run is constant" begin
        for (P0, Q0, V0, nP, nQ) in ((1.0, 0.0, 1.0, 0.0, 0.0), (0.8, 0.25, 1.02, 0.0, 0.0),
                                     (0.3, -0.1, 0.97, 1.0, 2.0))
            p = params_from_cons(merge(s1_cons, Dict("nP" => nP, "nQ" => nQ)))
            dt = 1 / 240
            s = GradPower.perc2_init(p, P0, Q0, V0)
            Ip0, Iq0 = GradPower.perc2_output(s)
            @test isfinite(Ip0) && isfinite(Iq0)
            # I = (Ip - jIq) V/|V|  =>  P = Ip*V, Q = Iq*V at V = V0
            @test abs(Ip0 * V0 - P0) < 1e-12
            @test abs(Iq0 * V0 - Q0) < 1e-12
            dmax = 0.0
            for k in 0:2400
                Ip, Iq = GradPower.perc2_step!(s, p, V0, 60.0, k * dt, dt)
                dmax = max(dmax, abs(Ip - Ip0), abs(Iq - Iq0))
            end
            @test dmax < 1e-12
            sn = GradPower.perc2_snapshot(s, V0)
            @test abs(sn.Vtf - V0) < 1e-12
            @test abs(sn.FracOn - 1) < 1e-12
            @test abs(sn.Ppu0 - 1) < 1e-12
            @test abs(sn.P - P0) < 1e-12 && abs(sn.Q - Q0) < 1e-12
        end
    end

    @testset "T11 zero-voltage input stays finite" begin
        dt = 1 / 240
        for cons in (s1_cons, merge(s1_cons, Dict("nP" => 2.0, "nQ" => 1.0)))
            p = params_from_cons(cons)
            s = GradPower.perc2_init(p, 1.0, 0.3, 1.0)
            allfinite = true
            for k in 0:480
                v = k < 240 ? 0.0 : 1.0     # 1 s at exactly 0 pu, then recover
                Ip, Iq = GradPower.perc2_step!(s, p, v, 60.0, k * dt, dt)
                sn = GradPower.perc2_snapshot(s, v)
                allfinite &= all(isfinite, (Ip, Iq, sn.Vtf, sn.ftf, sn.FracOn, sn.P, sn.Q, sn.Ppu0))
            end
            @test allfinite
        end
    end

    @testset "Tt floor = tt_floor_steps*dt" begin
        dt = 1 / 240
        # Tv tiny: Vtf follows V within one step, so the command jumps at
        # the step; the Tt lag then gives a one-step response with
        # time constant 2*dt: x1 = x0 + (u - x0)*(1 - exp(-dt/Tt)).
        base = merge(s1_cons, Dict("Tv" => 1e-9, "nP" => 0.0, "nQ" => 0.0))
        function run_tt(Tt; floor_steps = 2, nsteps = 1)
            p = GradPower.PERC2Params(; (Symbol(k) => (Symbol(k) in PERC2_FLAGS ? Bool(round(v) == 1) : Float64(v))
                                          for (k, v) in merge(base, Dict("Tt" => Tt)))...,
                                      tt_floor_steps = floor_steps)
            s = GradPower.perc2_init(p, 1.0, 0.0, 1.0)
            out = Tuple{Float64,Float64}[]
            for k in 0:nsteps
                GradPower.perc2_step!(s, p, k == 0 ? 1.0 : 1.25, 60.0, k * dt, dt)
                push!(out, GradPower.perc2_output(s))
            end
            return out
        end
        long = run_tt(0.001; nsteps = 2400)
        Ip0 = 1.0
        Ipf = long[end][1]
        @test abs(Ipf - 0.8) < 1e-6            # P0/Vtf with V = 1.25
        a = 1 - exp(-dt / (2dt))
        one = run_tt(0.001; nsteps = 1)
        @test abs(one[2][1] - (Ip0 + (Ipf - Ip0) * a)) < 1e-6
        # the floor makes Tt = 0.001 identical to an explicit Tt = 2dt
        ref = run_tt(2dt; nsteps = 50)
        got = run_tt(0.001; nsteps = 50)
        @test maximum(abs(r[1] - g[1]) for (r, g) in zip(ref, got)) < 1e-12
        # a larger tt_floor_steps lengthens it: Tt_eff = 4dt
        got4 = run_tt(0.001; floor_steps = 4, nsteps = 50)
        ref4 = run_tt(4dt; nsteps = 50)
        @test maximum(abs(r[1] - g[1]) for (r, g) in zip(ref4, got4)) < 1e-12
        @test abs(got4[2][1] - Ip0) < abs(got[2][1] - Ip0)
    end

    @testset "T7 Ldfact sets the ramp duration (Tramp*Ldfact)" begin
        dt = load_case("s2_vlm_dip_short").dt
        bps = load_case("s2_vlm_dip_short").breakpoints
        for Tramp in (1.25, 2.0), Ld in (0.3, 0.6, 0.8, 1.0)
            p = params_from_cons(merge(s2_cons, Dict("Ldfact" => Ld, "Tramp" => Tramp)))
            snaps = playback(p, bps, 4.0, dt)
            frac = [s.FracOn for s in snaps]
            ev = perc2_events_from_fracon(frac, dt)
            @test haskey(ev, :reconnect_end)
            dur = ev[:reconnect_end] - ev[:reconnect_start]
            @test abs(dur - Tramp * Ld) <= dt + 1e-9
            @test abs(frac[end] - (1 - p.fcease + p.fcease * p.frecon)) < 1e-9
        end
    end

    @testset "T12 validate rejects bad parameters" begin
        val = GradPower.validate
        @test val(GradPower.PERC2Params()) isa GradPower.PERC2Params
        for nm in (:Tt, :Tv, :Tf, :Tcease, :Tdelay, :Tramp, :Trecon, :TLrampL, :TLreconL,
                   :TLrampH, :TLreconH, :TVLC1, :TVLC2, :TVLC4, :TVHC1, :TVHC4, :Tvp, :Tvq)
            p = GradPower.PERC2Params(; nm => -0.1)
            @test_throws ArgumentError val(p)
        end
        @test_throws ArgumentError val(GradPower.PERC2Params(; Tt = NaN))
        @test_throws ArgumentError val(GradPower.PERC2Params(; Imax = 0.0))
        @test_throws ArgumentError val(GradPower.PERC2Params(; Ipmax = 0.1, Ipmin = 0.2))
        @test_throws ArgumentError val(GradPower.PERC2Params(; Kvp = 0.3, Tvp = 0.0))
        # every check, the long checks and the transient blocks are accepted
        @test val(GradPower.PERC2Params(; VHMflag = true, VLLflag = true, VLHflag = true,
                                          FLflag = true, FHflag = true)) isa GradPower.PERC2Params
    end

    @testset "Imax binds in a 0.6 pu sag" begin
        dt = 1 / 240
        bps = [(0.0, 1.0, 60.0), (0.1, 0.6, 60.0), (1.0, 1.0, 60.0)]
        for Ipmax in (1.0, 1.5)
            P0, Q0 = 1.0, 0.4
            # QPratio = Q0/P0 keeps all of Q on the dynamic (limited) path;
            # Iqextra (QPratio < Q0/P0) is a constant current outside the limits
            p = params_from_cons(merge(s2_cons, Dict("Imax" => 1.0, "Ldfact" => 0.8, "Ipmax" => Ipmax,
                                                     "QPratio" => Q0 / P0)))
            mbase = P0 / p.Ldfact
            s = GradPower.perc2_init(p, P0, Q0, 1.0)
            maxIp = 0.0; maxI = 0.0; viol = false
            for k in 0:round(Int, 1.5 / dt)
                t = k * dt
                v, f = perc2_profile(bps, t)
                Ip, Iq = GradPower.perc2_step!(s, p, v, f, t, dt)
                viol |= Ip > min(Ipmax, p.Imax) * mbase + 1e-12
                viol |= sqrt(Ip^2 + Iq^2) > p.Imax * mbase + 1e-12
                maxIp = max(maxIp, Ip); maxI = max(maxI, sqrt(Ip^2 + Iq^2))
            end
            @test !viol
            @test maxI > 0.99 * p.Imax * mbase       # the limit is actually reached
        end
    end

    # Closed-form discrete solutions (input held over each step, exact
    # exponential update; the step lands on sample k0):
    #   washout  sK/(1+sT), step dV:      y_j = (K/T) dV exp(-(j+1) dt/T)
    #   lead-lag (1+sTa)/(1+sTb), step d: y_j = u0 + d (1 - (1 - Ta/Tb) exp(-(j+1) dt/Tb))
    # nP = nQ = 1 make Ip = Pctrl*Mbase/V0 and Iq = Qctrl*Mbase/V0, so the
    # block outputs can be read from perc2_output; Tt = Tf = 0 and wide
    # limits remove every other dynamic.
    dt = 1 / 240
    k0 = 10; nsteps = 400
    TR_WIDE = Dict("nP" => 1.0, "nQ" => 1.0, "Tt" => 0.0, "Tf" => 0.0, "Tv" => 0.05,
                   "Ipmax" => 5.0, "Ipmin" => 0.0, "Iqmax" => 5.0, "Iqmin" => -5.0,
                   "Imax" => 20.0, "fcease" => 0.0, "Kdroop" => 8.0, "dbfl" => 0.0,
                   "dbfh" => 0.0, "Ldfact" => 0.8)
    function tr_params(extra)
        c = merge(s2_cons, TR_WIDE, Dict{String,Any}(string(k) => v for (k, v) in extra))
        p = params_from_cons(c)
        return GradPower.PERC2Params(; (n => getfield(p, n) for n in fieldnames(GradPower.PERC2Params))...,
                                     tt_floor_steps = 0)
    end
    P0, V0 = 1.0, 1.0
    mbase = P0 / 0.8
    # returns (pctrl[k], qctrl[k]) for k = 0..nsteps
    function tr_run(p; vfun = k -> V0, ffun = k -> 60.0)
        s = GradPower.perc2_init(p, P0, 0.0, V0)
        pc = Float64[]; qc = Float64[]
        for k in 0:nsteps
            v = vfun(k)
            GradPower.perc2_step!(s, p, v, ffun(k), k * dt, dt)
            Ip, Iq = GradPower.perc2_output(s)
            push!(pc, Ip * V0 / mbase); push!(qc, Iq * V0 / mbase)
        end
        return pc, qc
    end

    @testset "transients: disabled blocks leave the flat run untouched" begin
        pc, qc = tr_run(tr_params(Dict()))
        @test maximum(abs, pc .- 0.8) < 1e-12 && maximum(abs, qc) < 1e-12
    end

    @testset "transients: P and Q washouts: step response" begin
        dV = 0.05
        vstep(k) = k >= k0 ? V0 + dV : V0
        for (K, T) in ((0.4, 0.1), (-0.25, 0.3), (1.0, 0.02))
            pc, _ = tr_run(tr_params(Dict(:Kvp => K, :Tvp => T)); vfun = vstep)
            _, qc = tr_run(tr_params(Dict(:Kvq => K, :Tvq => T)); vfun = vstep)
            exp_y(k) = k < k0 ? 0.0 : (K / T) * dV * exp(-(k - k0 + 1) * dt / T)
            @test maximum(abs(pc[k+1] - (0.8 + exp_y(k))) for k in 0:nsteps) < 1e-11
            @test maximum(abs(qc[k+1] - exp_y(k)) for k in 0:nsteps) < 1e-11
        end
        # sign: a voltage rise raises the P request for Kvp > 0, a fall lowers it
        pc, _ = tr_run(tr_params(Dict(:Kvp => 0.4, :Tvp => 0.1)); vfun = k -> k >= k0 ? 0.95 : 1.0)
        @test pc[k0+1] < 0.8
        # constant voltage at a different level: the washout decays to zero
        pc, _ = tr_run(tr_params(Dict(:Kvp => 0.4, :Tvp => 0.05)); vfun = vstep)
        @test abs(pc[end] - 0.8) < 1e-3
        # the P washout does not touch Q and vice versa
        _, qc = tr_run(tr_params(Dict(:Kvp => 0.4, :Tvp => 0.1)); vfun = vstep)
        @test maximum(abs, qc) < 1e-12
    end

    @testset "transients: P lead-lag: step response (frequency-droop step)" begin
        # Tf = 0: ftf follows f at once; step 60 -> 60.3 Hz: d = Kdroop*0.3/60
        d = 8.0 * 0.3 / 60
        fstep(k) = k >= k0 ? 60.3 : 60.0
        for (Ta, Tb) in ((0.5, 0.2), (0.05, 0.3), (0.0, 0.25), (0.3, 0.1))
            pc, qc = tr_run(tr_params(Dict(:Tap => Ta, :Tbp => Tb)); ffun = fstep)
            y(k) = k < k0 ? 0.8 : 0.8 + d * (1 - (1 - Ta / Tb) * exp(-(k - k0 + 1) * dt / Tb))
            @test maximum(abs(pc[k+1] - y(k)) for k in 0:nsteps) < 1e-11
            @test maximum(abs, qc) < 1e-12
        end
        # bypassed forms: Tb = 0 (Ta must be 0 too), Ta == Tb
        for (Ta, Tb) in ((0.0, 0.0), (0.4, 0.4))
            pc, _ = tr_run(tr_params(Dict(:Tap => Ta, :Tbp => Tb)); ffun = fstep)
            @test maximum(abs(pc[k+1] - (k < k0 ? 0.8 : 0.8 + d)) for k in 0:nsteps) < 1e-12
        end
        # initial value: the lead-lag starts at its steady state, no start-up transient
        pc, _ = tr_run(tr_params(Dict(:Tap => 0.5, :Tbp => 0.2)))
        @test maximum(abs, pc .- 0.8) < 1e-12
    end

    @testset "transients: Q lead-lag on a washout transient" begin
        # Q request = LL_Q(washout): the washout output is the sampled
        # exponential w_j above; the lead-lag is run on that sequence.
        dV, K, Tw, Ta, Tb = 0.05, 0.3, 0.2, 0.4, 0.15
        vstep(k) = k >= k0 ? V0 + dV : V0
        _, qc = tr_run(tr_params(Dict(:Kvq => K, :Tvq => Tw, :Taq => Ta, :Tbq => Tb)); vfun = vstep)
        w(k) = k < k0 ? 0.0 : (K / Tw) * dV * exp(-(k - k0 + 1) * dt / Tw)
        a = 1 - exp(-dt / Tb)
        x = 0.0; err = 0.0
        for k in 0:nsteps
            x += (w(k) - x) * a
            err = max(err, abs(qc[k+1] - (x + (Ta / Tb) * (w(k) - x))))
        end
        @test err < 1e-11
    end

    @testset "transients: P washout + lead-lag + droop combined" begin
        dV, K, Tw, Ta, Tb = 0.04, 0.5, 0.1, 0.2, 0.6
        d = 8.0 * 0.1 / 60
        vstep(k) = k >= k0 ? V0 + dV : V0
        fstep(k) = k >= k0 + 20 ? 60.1 : 60.0
        pc, _ = tr_run(tr_params(Dict(:Kvp => K, :Tvp => Tw, :Tap => Ta, :Tbp => Tb));
                       vfun = vstep, ffun = fstep)
        u(k) = 0.8 + (k >= k0 + 20 ? d : 0.0) + (k < k0 ? 0.0 : (K / Tw) * dV * exp(-(k - k0 + 1) * dt / Tw))
        a = 1 - exp(-dt / Tb)
        x = 0.8; err = 0.0
        for k in 0:nsteps
            x += (u(k) - x) * a
            err = max(err, abs(pc[k+1] - (x + (Ta / Tb) * (u(k) - x))))
        end
        @test err < 1e-11
    end

    @testset "transients: transients are clipped by the current limits" begin
        # a large washout kick above Ipmax must saturate at Ipmax*Mbase
        p = tr_params(Dict(:Kvp => 5.0, :Tvp => 0.05, :Ipmax => 1.0, :Imax => 20.0))
        s = GradPower.perc2_init(p, P0, 0.0, V0)
        mx = 0.0
        for k in 0:200
            GradPower.perc2_step!(s, p, k >= 5 ? 1.1 : 1.0, 60.0, k * dt, dt)
            mx = max(mx, GradPower.perc2_output(s)[1])
        end
        @test abs(mx - 1.0 * mbase) < 1e-12
    end

    @testset "transients: disabled blocks are bit-identical to the plain run" begin
        variants = (Dict("Tvp" => 0.3, "Tvq" => 0.01),                 # Kvp = Kvq = 0: Tvp/Tvq irrelevant
                    Dict("Tap" => 0.7, "Tbp" => 0.7, "Taq" => 0.2, "Tbq" => 0.2),   # Ta == Tb
                    Dict("Tap" => 0.0, "Tbp" => 0.0, "Taq" => 0.0, "Tbq" => 0.0))   # Tb = 0 bypass
        fields = (:Vtf, :ftf, :FracOn, :P, :Q, :Ppu0, :ip_m, :iq_m)
        for n in names
            c = load_case(n)
            base = playback(params_from_cons(c.cons), c)
            for var in variants
                got = playback(params_from_cons(merge(c.cons, var)), c)
                same = all(all(getfield(a, f) === getfield(b, f) || getfield(a, f) == getfield(b, f)
                               for f in fields) for (a, b) in zip(base, got))
                same || @info "PERC2 disabled-block variant differs" case = n variant = var
                @test same
            end
        end
    end

    @testset "transients: validate accepts washouts and lead-lags" begin
        pw = GradPower.PERC2Params(; Kvp = 0.3)
        @test GradPower.validate(pw) === pw
        pl = GradPower.PERC2Params(; Tap = 0.2, Tbp = 0.5)
        @test GradPower.validate(pl) === pl
        @test_throws ArgumentError GradPower.validate(GradPower.PERC2Params(; Tap = 0.2, Tbp = 0.0))
    end

    @testset "QPratio: Iqextra is a constant current that follows FracOn" begin
        P0, Q0, V0 = 1.0, 0.4, 1.02
        dt = 1 / 240
        mk(; kw...) = (d = merge(s2_cons, Dict{String,Any}(string(k) => v for (k, v) in kw));
                       p = params_from_cons(d);
                       GradPower.PERC2Params(; (n => getfield(p, n) for n in fieldnames(GradPower.PERC2Params))...))
        flat(p) = (s = GradPower.perc2_init(p, P0, Q0, V0); (s, GradPower.perc2_snapshot(s, V0)))
        # Iqextra = (Q0 - QPratio*P0)/V0, flat total Iq = Q0/V0 in every split
        for qp in (0.0, 0.1, 0.25, 0.4)
            s, sn = flat(mk(; QPratio = qp))
            @test abs(sn.Iqextra - (Q0 - qp * P0) / V0) < 1e-12
            Ip, Iq = GradPower.perc2_output(s)
            @test abs(Iq * V0 - Q0) < 1e-12 && abs(Ip * V0 - P0) < 1e-12
        end
        # QPratio = Q0/P0: no constant-current part
        s, sn = flat(mk(; QPratio = Q0 / P0))
        @test abs(sn.Iqextra) < 1e-15
        # use_qpratio = false: the whole Q goes through the dynamic path
        p = GradPower.PERC2Params(; (n => getfield(mk(), n) for n in fieldnames(GradPower.PERC2Params) if n !== :use_qpratio)...,
                                  use_qpratio = false)
        s, sn = flat(p)
        @test abs(sn.Iqextra) < 1e-15
        # QPratio = Q0/P0 and use_qpratio = false behave identically on a dip
        bps = load_case("s2_vlm_dip_short").breakpoints
        a = playback(mk(; QPratio = Q0 / P0), bps, 2.0, dt; P0 = P0, Q0 = Q0)
        b = playback(p, bps, 2.0, dt; P0 = P0, Q0 = Q0)
        @test maximum(abs(x.iq_m - y.iq_m) for (x, y) in zip(a, b)) < 1e-12

        # QPratio = 0: during the cease the output Iq is Iqextra*FracOn,
        # through the Tt lag (first-order, Tt_eff = max(Tt, tt_floor*dt))
        for Tt in (0.0, 0.02)
            pq = mk(; QPratio = 0.0, Tt = Tt)
            snaps = playback(pq, bps, 3.0, dt; P0 = P0, Q0 = Q0)
            mbase = P0 / pq.Ldfact
            iqx = (Q0 - 0.0 * P0) / bps[1][2]      # playback initialises at V0 = v(0)
            @test minimum(x.FracOn for x in snaps) < 0.2          # it really ceases
            Iq = [x.iq_m * mbase for x in snaps]
            if Tt == 0.0 && pq.tt_floor_steps == 0
                @test maximum(abs(Iq[k] - iqx * snaps[k].FracOn) for k in eachindex(snaps)) < 1e-12
            else
                a = 1 - exp(-dt / max(Tt, pq.tt_floor_steps * dt))
                x = Iq[1]; err = abs(x - iqx)
                for k in 2:length(snaps)
                    x += (iqx * snaps[k].FracOn - x) * a
                    err = max(err, abs(Iq[k] - x))
                end
                @test err < 1e-11
            end
            # the cease drives the current down and the ramp back to its final value
            kmin = argmin([x.FracOn for x in snaps])
            @test Iq[kmin+3] < 0.5 * iqx
            @test abs(Iq[end] - iqx * snaps[end].FracOn) < 1e-9
            # Iqextra bypasses nQ and the current limits: nQ and Iqmax do not change it
            pn = mk(; QPratio = 0.0, Tt = Tt, nQ = 2.0, Iqmax = 0.05, Iqmin = -0.05)
            sn2 = playback(pn, bps, 3.0, dt; P0 = P0, Q0 = Q0)
            @test maximum(abs(x.iq_m - y.iq_m) for (x, y) in zip(snaps, sn2)) < 1e-12
        end
    end
end
