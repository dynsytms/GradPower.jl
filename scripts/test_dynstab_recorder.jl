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

    # check_self_stability uses the streaming recorder; its separation must
    # equal stability_metrics on the full kicked trajectory.
    empty!(sys.dynamic.events)
    chk = check_self_stability(sys, dprob, z0; t_final=2.0, dt=dt)
    gptr, _ = generator_pointers(sys)
    dprob.zvec .= z0; dprob.zvec[gptr[1] + 4] += 1e-6
    _, traj = GradPower.integrate!(dprob, sys, 2.0; dt=dt)
    @test chk.sep_deg == stability_metrics(sys, traj)["max_angle_sep_deg"]
end
