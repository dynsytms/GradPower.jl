# PERC2: .dyr import of PERCBL2U records and batched (multi-scenario)
# integration. Uses examples/ieee9_perc2.dyr (default PERC2 parameters).

using Test
using GradPower

@testset "PERC2 .dyr import and batched" begin
    ex = joinpath(@__DIR__, "..", "examples")
    raw = joinpath(ex, "ieee9_v33.raw")
    dyr = joinpath(ex, "ieee9_perc2.dyr")

    @testset "record parsing" begin
        recs = GradPower.read_perc2_dyr_records(dyr)
        @test length(recs) == 1
        r = recs[1]
        @test (r.bus, r.id, r.component, r.description, r.load_fraction) == (5, "1", 1, "Data center", 0.8)
        # the example record holds the default PERC2 parameters
        @test r.params == GradPower.PERC2Params()
        @test length(GradPower.PERC2_DYR_CON_NAMES) == 57

        txt = read(dyr, String)
        mktemp() do p, io
            close(io)
            # 'USRLOD' variant, no MAINxx10 record -> whole load with a warning
            write(p, replace(replace(txt, "'USRLODC'" => "'USRLOD'"),
                             r"5 'MAINBL10'[^/]*/\n" => "", r"5 'LSTCBL'[^/]*/\n" => ""))
            r2 = GradPower.read_perc2_dyr_records(p)[1]
            @test r2.load_fraction === nothing
            @test (@test_logs (:warn, r"no matching MAINxx10") GradPower._perc2_dyr_fraction(r2)) == 1.0
            # non-default CON lands in the right field
            write(p, replace(txt, "0.7 -0.7 1.5 0.9 0.5 1.2" => "0.7 -0.7 1.5 0.9 0.45 1.2"))
            @test GradPower.read_perc2_dyr_records(p)[1].params.VLcease == 0.45
            write(p, replace(txt, " 62 100 /" => " 62 /"))
            @test_throws ArgumentError GradPower.read_perc2_dyr_records(p)
            write(p, replace(txt, "'PERCBL2U' 230 1" => "'PERCZN2U' 230 3"))
            @test_throws ArgumentError GradPower.read_perc2_dyr_records(p)
        end
    end

    @testset "coverage" begin
        @test "PERCBL2U" in native_dyr_models()
        rep = analyze_dyr_coverage(raw, dyr)
        st = Dict(x.source_model => x.status for x in rep.records)
        @test st["PERCBL2U"] == :native
        @test st["MAINBL10"] == :native
        @test st["LSTCBL"] == :unsupported
    end

    function build(; dyr_import::Bool, fault_bus::Int, rf::Float64)
        if dyr_import
            ps = from_psse(raw, dyr)
            GradPower.build_network!(ps); GradPower.runpf!(ps)
        else
            ps = from_psse(raw, joinpath(ex, "ieee9bus_gov.dyr"))
            GradPower.build_network!(ps); GradPower.runpf!(ps)
            add_perc2_load!(ps, 5, "1"; fraction = 0.8,
                            params = GradPower.PERC2Params(VLcease = 0.7, VLrecon = 0.8, Tcease = 0.02))
        end
        dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
        GradPower.add_event!(ps, GradPower.ContingencyEvent(ps.busmap[fault_bus], rf, 0.1, 0.175))
        return ps, dp
    end

    @testset "from_psse equals manual add_perc2_load!" begin
        ps1 = from_psse(raw, dyr)
        ps2 = from_psse(raw, joinpath(ex, "ieee9bus_gov.dyr"))
        for ps in (ps1, ps2)
            GradPower.build_network!(ps); GradPower.runpf!(ps)
        end
        add_perc2_load!(ps2, 5, "1"; fraction = 0.8)
        dp1 = GradPower.DynamicProblem(ps1); GradPower.initialize_dynamics!(dp1, ps1)
        dp2 = GradPower.DynamicProblem(ps2); GradPower.initialize_dynamics!(dp2, ps2)
        @test dp1.pvec == dp2.pvec && dp1.zvec == dp2.zvec
        _, x1 = integrate!(dp1, ps1, 2.0)
        _, x2 = integrate!(dp2, ps2, 2.0)
        @test x1 == x2
        @test maximum(abs.(x1 .- x1[:, 1])) < 1e-6        # flat run
    end

    @testset "batched == integrate! per scenario ($label, $integrator)" for
            (label, kw) in (("dyr, fault at bus 5", (dyr_import = true, fault_bus = 5, rf = 0.01)),
                            ("VLcease 0.7, fault at bus 4", (dyr_import = false, fault_bus = 4, rf = 0.02))),
            integrator in (:backward_euler, :crank_nicolson)
        ps, dp = build(; kw...)
        dyn = ps.dynamic
        M = 3
        bl = GradPower.BatchedLayout(dp, ps, M)
        hp = dyn.devices[findfirst(d -> d.dtype isa GradPower.Genrou, dyn.devices)].par_ptr + 6
        pp = dyn.devices[findfirst(d -> d.dtype isa PERC2Load, dyn.devices)].par_ptr
        for m in 1:M
            bl.p[m, hp] *= 1.0 + 0.15 * (m - 1)          # Genrou H
            bl.p[m, pp + 2] *= 1.0 - 0.05 * (m - 1)      # PERC2 P0 (own perc2_init)
        end
        p_in, z_in = copy(bl.p), copy(bl.z)
        tf = 2.0
        out = Dict{Bool,Any}()
        for threaded in (false, true)
            bl.z .= z_in
            out[threaded] = GradPower.integrate_batched!(bl, ps, tf; integrator, threaded)
        end
        @test bl.p == p_in
        for m in 1:M
            dpm = deepcopy(dp)
            dpm.pvec .= p_in[m, :]; dpm.zvec .= z_in[m, :]
            log = Any[]
            t, x = integrate!(dpm, ps, tf; integrator, perc2_log = log)
            @test any(e -> e.FracOn < 0.5, log)          # the PERC2 load ceased
            for threaded in (false, true)
                tb, xb = out[threaded]
                @test tb == t
                @test maximum(abs.(xb[m] .- x)) <= 1e-10
            end
        end
        @test maximum(abs.(out[false][2][1] .- out[false][2][3])) > 1e-4
    end
end
