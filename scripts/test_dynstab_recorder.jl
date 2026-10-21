# Equivalence tests for the streaming data-generation path (feat/recorder).
#
#   julia --project=scripts scripts/test_dynstab_recorder.jl
#
# `simulate_sample(...; output_mode=:derived_channels)` (stores only the
# downsampled channel rows, labels from a step callback) must give exactly the
# channels and labels of the original
# full-trajectory path: `dynamics_channels(sys, traj; downsample)` and
# `stability_metrics(sys, traj)`. Checked bitwise on stable and unstable
# scenarios, downsample factors that do and do not divide the step count, with
# and without faults.

using Test
using GradPower
include(joinpath(@__DIR__, "dynstab_io.jl"))

function full_path(dprob, sys, t_final; dt, downsample, kw...)
    tv, traj = GradPower.integrate!(dprob, sys, t_final; dt=dt)
    ch = Dict{String,Any}(dynamics_channels(sys, traj; downsample=downsample))
    ch["time"] = tv[1:downsample:end]
    ch["output_mode"] = "derived_channels"
    return ch, stability_metrics(sys, traj; kw...)
end

same_dict(a, b) = keys(a) == keys(b) && all(isequal(a[k], b[k]) for k in keys(a))

@testset "dynstab streaming path == full-trajectory path" begin
    case = Dict("raw" => "examples/IEEE39.raw", "dyr" => "examples/IEEE39_gov.dyr")
    sys, _ = build_system(case)
    dprob, _ = initialized_problem(sys)
    z0 = copy(dprob.zvec)
    dt = 1/120
    labels = Bool[]
    for (bus, rf, dur) in ((16, 0.01, 0.1), (16, 0.01, 0.5), (1, 0.05, 0.3), (0, 0.0, 0.0)),
        ds in (1, 2, 3, 7), tf in (1.5, 2.0)
        empty!(sys.dynamic.events)
        bus > 0 && GradPower.add_event!(sys, GradPower.ContingencyEvent(sys.busmap[bus], rf, 0.2, 0.2 + dur))
        kw = (angle_sep_threshold=deg2rad(180.0), tail_fraction=0.2, decay_ratio=0.5)
        dprob.zvec .= z0
        ch_f, m_f = full_path(dprob, sys, tf; dt=dt, downsample=ds, kw...)
        zf_f = copy(dprob.zvec)
        dprob.zvec .= z0
        ch_s, m_s = simulate_sample(dprob, sys, tf; dt=dt, downsample=ds,
                                    output_mode=:derived_channels, kw...)
        @test same_dict(ch_s, ch_f)
        @test same_dict(m_s, m_f)
        @test dprob.zvec == zf_f
        push!(labels, m_f["stable"])
    end
    @test any(labels) && !all(labels)          # both classes covered

    # Early stop on loss of synchronism: same `stable` label as the full run,
    # channels = prefix of the full run's channels, stops only unstable runs.
    nstopped = 0
    nroundtrips = 0
    # last two lose synchronism (separation ~6000 / ~4300 deg over 3 s)
    for (bus, rf, dur) in ((16, 0.01, 0.1), (1, 0.05, 0.3), (26, 0.01, 0.4),
                           (16, 0.001, 0.6), (26, 0.001, 0.6)),
        ds in (1, 3), thr_deg in (180.0, 360.0),
        mode in (:all_states, :derived_channels)
        empty!(sys.dynamic.events)
        GradPower.add_event!(sys, GradPower.ContingencyEvent(sys.busmap[bus], rf, 0.2, 0.2 + dur))
        kw = (angle_sep_threshold=deg2rad(180.0), tail_fraction=0.2, decay_ratio=0.5)
        dprob.zvec .= z0
        ch_f, m_f = simulate_sample(dprob, sys, 3.0; dt=dt, downsample=ds,
                                    output_mode=mode, kw...)
        dprob.zvec .= z0
        ch_e, m_e = simulate_sample(dprob, sys, 3.0; dt=dt, downsample=ds,
                                    output_mode=mode, early_stop_sep=deg2rad(thr_deg), kw...)
        @test m_e["stable"] == m_f["stable"]
        T = ch_e["T"]
        series = mode === :all_states ? ("states",) : ("bus_vm", "bus_va", "gen_delta", "gen_omega")
        for key in series
            @test ch_e[key] == ch_f[key][:, 1:T]
        end
        @test ch_e["time"] == ch_f["time"][1:T]
        if m_e["terminated_early"]
            nstopped += 1
            @test !m_f["stable"]
            @test m_e["stop_reason"] == "angle_separation"
            @test m_e["t_stop"] < 3.0
            @test m_f["max_angle_sep_deg"] >= thr_deg
            @test m_e["max_angle_sep_deg"] > thr_deg
            @test (T - 1) * ds * dt <= m_e["t_stop"] + 1e-12
            # The stored last time is an actual saved step, not necessarily
            # the (off-stride) termination time.
            @test ch_e["time"][end] <= m_e["t_stop"] + 1e-12
            if mode === :all_states && nroundtrips == 0
                sc = Scenario(1, 1.0, 0, sys.busmap[bus], bus, rf, 0.2, 0.2 + dur)
                mktempdir() do dir
                    path = joinpath(dir, "stopped.h5")
                    h5open(path, "w") do fid
                        write_sample!(fid, 1, sc, ch_e, m_e)
                    end
                    h5open(path, "r") do fid
                        sample = fid["samples/000001"]
                        @test read(sample["states"]) == ch_e["states"]
                        @test read(sample["time"]) == ch_e["time"]
                        @test same_dict(read(fid["state_schema"]), ch_e["state_schema"])
                        @test read(attributes(sample)["terminated_early"]) == 1
                        @test read(attributes(sample)["t_stop"]) == m_e["t_stop"]
                    end
                end
                nroundtrips += 1
            end
        else
            @test m_e["stop_reason"] == "none" && T == ch_f["T"]
            @test all(isequal(m_e[k], m_f[k]) for k in keys(m_f))
        end
    end
    @test nstopped > 0
    @test nroundtrips == 1
    @test_throws ErrorException simulate_sample(dprob, sys, 1.0; dt=dt, early_stop_sep=deg2rad(90.0))

    # check_self_stability uses the streaming recorder; its separation must
    # equal stability_metrics on the full kicked trajectory.
    empty!(sys.dynamic.events)
    chk = check_self_stability(sys, dprob, z0; t_final=2.0, dt=dt)
    gptr, _ = generator_pointers(sys)
    dprob.zvec .= z0; dprob.zvec[gptr[1] + 4] += 1e-6
    _, traj = GradPower.integrate!(dprob, sys, 2.0; dt=dt)
    @test chk.sep_deg == stability_metrics(sys, traj)["max_angle_sep_deg"]
end

@testset "all-state default and HDF5 roundtrip" begin
    @test parse_output_mode() === :all_states
    @test parse_output_mode("all_states") === :all_states
    @test parse_output_mode(:derived_channels) === :derived_channels
    @test_throws ArgumentError parse_output_mode("typo")

    case = Dict("raw" => "examples/ieee9_v33.raw", "dyr" => "examples/ieee9bus_gov.dyr")
    sys, _ = build_system(case)
    dp, _ = initialized_problem(sys)
    z0 = copy(dp.zvec)
    dt = 1/120
    GradPower.add_event!(sys, GradPower.ContingencyEvent(4, 0.02, 0.1, 0.2))
    tv, traj = GradPower.integrate!(dp, sys, 0.5; dt=dt)
    zf = copy(dp.zvec)
    full_metrics = stability_metrics(sys, traj)
    sc = Scenario(1, 1.0, 0, 4, sys.buses[4].i, 0.02, 0.1, 0.2)

    for ds in (1, 7, 100)
        dp.zvec .= z0
        ch, m = simulate_sample(dp, sys, 0.5; dt=dt, downsample=ds)
        @test ch["output_mode"] == "all_states"
        @test ch["states"] == traj[:, 1:ds:end]
        @test eltype(ch["states"]) === Float64
        @test ch["time"] == tv[1:ds:end]
        @test ch["T"] == length(ch["time"])
        @test dp.zvec == zf
        @test same_dict(m, full_metrics)
        @test !haskey(ch, "bus_vm")
        schema = ch["state_schema"]
        @test schema["index"] == collect(1:length(z0))
        @test all(v -> length(v) == length(z0), values(schema))
        @test all(v -> eltype(v) in (Int, String, Bool), values(schema))
        @test count(==("differential"), schema["kind"]) == sys.dynamic.diff_dim
        @test count(==("algebraic"), schema["kind"]) == sys.dynamic.alg_dim + 2length(sys.buses)
        @test Set(schema["bus"]) == Set(b.i for b in sys.buses)
        @test count(==("bus"), schema["model"]) == 2length(sys.buses)

        mktempdir() do dir
            h5open(joinpath(dir, "all.h5"), "w") do fid
                write_sample!(fid, 1, sc, ch, m)
                write_sample!(fid, 2, sc, ch, m) # shared row schema
                @test length(keys(fid["state_schema"])) == 7
                changed_schema = deepcopy(schema)
                changed_schema["device_id"][1] *= "-different"
                @test_throws ArgumentError write_state_schema!(fid, changed_schema)
            end
            h5open(joinpath(dir, "all.h5"), "r") do fid
                stored = read(fid["samples/000001"])
                @test stored["states"] == ch["states"]
                @test eltype(stored["states"]) === Float64
                @test stored["time"] == ch["time"]
                saved_schema = read(fid["state_schema"])
                @test same_dict(saved_schema, schema)
                @test eltype(read(fid["state_schema/online"])) === Bool
                @test read(attributes(fid["samples/000001"])["output_mode"]) == "all_states"
                @test read(attributes(fid["samples/000001"])["T"]) == length(ch["time"])
                @test read(attributes(fid["samples/000001"])["stable"]) == Int(m["stable"])

                @testset "named-accessor roundtrip, downsample=$ds" begin
                    channels = StateChannel[
                        StateChannel(saved_schema["index"][i], Symbol(saved_schema["model"][i]),
                                     saved_schema["bus"][i], saved_schema["device_id"][i],
                                     Symbol(saved_schema["name"][i]), Symbol(saved_schema["kind"][i]),
                                     saved_schema["online"][i])
                        for i in eachindex(saved_schema["index"])
                    ]
                    result = SimulationResult(stored["time"], stored["states"], channels)
                    cols = 1:ds:size(traj, 2)
                    derived = dynamics_channels(sys, traj; downsample=ds)
                    @test result.times == tv[cols]
                    for c in channels
                        @test state(result, c.model, c.bus, c.device_id, c.name) == traj[c.index, cols]
                    end

                    gptr, _ = generator_pointers(sys)
                    machines = sort!(unique([(c.bus, c.device_id) for c in channels if c.model === :genrou]))
                    for (g, p) in enumerate(gptr)
                        c = only(filter(c -> c.index == p + 4, channels))
                        k = findfirst(==((c.bus, c.device_id)), machines)
                        @test speed(result, c.bus, c.device_id) == traj[p + 4, cols]
                        @test speed(result, k) == traj[p + 4, cols]
                        @test state(result, :genrou, k, :delta) == traj[p + 5, cols]
                        @test Float32.(speed(result, k)) == derived["gen_omega"][g, :]
                        @test Float32.(state(result, :genrou, k, :delta)) == derived["gen_delta"][g, :]
                    end

                    net = sys.dynamic.diff_dim + sys.dynamic.alg_dim
                    for (b, bus) in enumerate(sys.buses)
                        vr = traj[net + 2(b - 1) + 1, cols]
                        vi = traj[net + 2(b - 1) + 2, cols]
                        @test bus_voltage(result, bus.i; component=:real) == vr
                        @test bus_voltage(result, bus.i; component=:imag) == vi
                        @test bus_voltage(result, bus.i) == complex.(vr, vi)
                        vm = bus_voltage(result, bus.i; component=:magnitude)
                        va = bus_voltage(result, bus.i; component=:angle)
                        @test vm == hypot.(vr, vi)
                        @test va == atan.(vi, vr)
                        @test Float32.(vm) == derived["bus_vm"][b, :]
                        @test Float32.(va) == derived["bus_va"][b, :]
                    end
                end
            end
        end
    end

    dp.zvec .= z0
    ch, m = simulate_sample(dp, sys, 0.5; dt=dt, downsample=7,
                             output_mode="derived_channels")
    ref = dynamics_channels(sys, traj; downsample=7)
    @test all(k -> isequal(ch[k], ref[k]), keys(ref))
    @test same_dict(m, full_metrics)
    @test !haskey(ch, "states") && !haskey(ch, "state_schema")
    mktempdir() do dir
        h5open(joinpath(dir, "compact.h5"), "w") do fid
            write_sample!(fid, 1, sc, ch, m)
        end
        h5open(joinpath(dir, "compact.h5"), "r") do fid
            @test !haskey(fid, "state_schema")
            stored = read(fid["samples/000001"])
            @test stored["time"] == tv[1:7:end]
            for key in ("bus_vm", "bus_va", "gen_delta", "gen_omega")
                @test stored[key] == ch[key]
                @test eltype(stored[key]) === Float32
            end
        end
    end
    @test_throws ArgumentError simulate_sample(dp, sys, 0.5; dt=dt, output_mode=:typo)
end
