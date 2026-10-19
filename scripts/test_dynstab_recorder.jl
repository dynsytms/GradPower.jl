# Equivalence tests for the streaming data-generation path (feat/recorder).
#
#   julia --project=scripts scripts/test_dynstab_recorder.jl
#
# `simulate_sample` (stores only the downsampled channel rows, labels from a
# step callback) must give exactly the channels and labels of the original
# full-trajectory path: `dynamics_channels(sys, traj; downsample)` and
# `stability_metrics(sys, traj)`. Checked bitwise on stable and unstable
# scenarios, downsample factors that do and do not divide the step count, with
# and without faults.

using Test
using GradPower
include(joinpath(@__DIR__, "dynstab_io.jl"))

function full_path(dprob, sys, t_final; dt, downsample, kw...)
    _, traj = GradPower.integrate!(dprob, sys, t_final; dt=dt)
    return dynamics_channels(sys, traj; downsample=downsample), stability_metrics(sys, traj; kw...)
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
        ch_s, m_s = simulate_sample(dprob, sys, tf; dt=dt, downsample=ds, kw...)
        @test same_dict(ch_s, ch_f)
        @test same_dict(m_s, m_f)
        @test dprob.zvec == zf_f
        push!(labels, m_f["stable"])
    end
    @test any(labels) && !all(labels)          # both classes covered

    # Early stop on loss of synchronism: same `stable` label as the full run,
    # channels = prefix of the full run's channels, stops only unstable runs.
    nstopped = 0
    # last two lose synchronism (separation ~6000 / ~4300 deg over 3 s)
    for (bus, rf, dur) in ((16, 0.01, 0.1), (1, 0.05, 0.3), (26, 0.01, 0.4),
                           (16, 0.001, 0.6), (26, 0.001, 0.6)),
        ds in (1, 3), thr_deg in (180.0, 360.0)
        empty!(sys.dynamic.events)
        GradPower.add_event!(sys, GradPower.ContingencyEvent(sys.busmap[bus], rf, 0.2, 0.2 + dur))
        kw = (angle_sep_threshold=deg2rad(180.0), tail_fraction=0.2, decay_ratio=0.5)
        dprob.zvec .= z0
        ch_f, m_f = simulate_sample(dprob, sys, 3.0; dt=dt, downsample=ds, kw...)
        dprob.zvec .= z0
        ch_e, m_e = simulate_sample(dprob, sys, 3.0; dt=dt, downsample=ds,
                                    early_stop_sep=deg2rad(thr_deg), kw...)
        @test m_e["stable"] == m_f["stable"]
        T = ch_e["T"]
        for key in ("bus_vm", "bus_va", "gen_delta", "gen_omega")
            @test ch_e[key] == ch_f[key][:, 1:T]
        end
        if m_e["terminated_early"]
            nstopped += 1
            @test !m_f["stable"]
            @test m_e["stop_reason"] == "angle_separation"
            @test m_e["t_stop"] < 3.0
            @test m_f["max_angle_sep_deg"] >= thr_deg
            @test m_e["max_angle_sep_deg"] > thr_deg
            @test (T - 1) * ds * dt <= m_e["t_stop"] + 1e-12
        else
            @test m_e["stop_reason"] == "none" && T == ch_f["T"]
            @test all(isequal(m_e[k], m_f[k]) for k in keys(m_f))
        end
    end
    @test nstopped > 0
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
