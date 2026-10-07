#!/usr/bin/env julia
# Phase 8 Remediation G2 -- Multi-fault scenario
#
# Two faults on different buses at different times, simulation completes
# with physically reasonable trajectories.
# Uses the IEEE 9-bus system. Fault 1: bus 7, ton=0.2, toff=0.3.
# Fault 2: bus 5, ton=0.5, toff=0.6.
#
# Pass criteria:
#   1. Integration completes without error.
#   2. Generator speeds deviate from initial during faults (not flat-line).
#   3. Fault 2 has a distinct effect (speed trajectories differ from
#      single-fault case after t=0.5).
#
# Writes artifacts/phase8/g2.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase8", "g2.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function run_multi_fault()
    ps = GradPower.from_psse(
        joinpath(REPO, "examples", "ieee9_v33.raw"),
        joinpath(REPO, "examples", "ieee9bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for dev in ps.dynamic.devices
        if dev.dtype isa GradPower.ZIPLoad; dev.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    # Fault 1: bus 7 (internal), ton=0.2, toff=0.3
    fault_bus1 = ps.busmap[7]
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus1, 0.02, 0.2, 0.3))
    # Fault 2: bus 5 (internal), ton=0.5, toff=0.6
    fault_bus2 = ps.busmap[5]
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus2, 0.02, 0.5, 0.6))

    tvec, traj = GradPower.integrate!(dp, ps, 2.0; dt=1.0/120.0)
    return ps, tvec, traj
end

function run_single_fault()
    ps = GradPower.from_psse(
        joinpath(REPO, "examples", "ieee9_v33.raw"),
        joinpath(REPO, "examples", "ieee9bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for dev in ps.dynamic.devices
        if dev.dtype isa GradPower.ZIPLoad; dev.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    fault_bus1 = ps.busmap[7]
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus1, 0.02, 0.2, 0.3))
    tvec, traj = GradPower.integrate!(dp, ps, 2.0; dt=1.0/120.0)
    return ps, tvec, traj
end

function main()
    t_start = time()
    criteria = Any[]
    all_passed = true

    # Criterion 1: multi-fault completes
    ok_run = false
    ps_multi = nothing; traj_multi = nothing
    try
        ps_multi, tvec_multi, traj_multi = run_multi_fault()
        ok_run = true
    catch e
        println("Multi-fault run failed: ", sprint(showerror, e, catch_backtrace()))
    end
    push!(criteria, Dict("name" => "multi_fault_completes",
                         "value" => ok_run, "threshold" => true, "passed" => ok_run))
    if !ok_run; all_passed = false; end

    # Criterion 2: speeds deviate during faults (not flat-line)
    ok_deviate = false
    max_speed_dev = 0.0
    if ok_run
        speed_idxs = GradPower.gen_speeds(ps_multi)
        for si in speed_idxs
            max_speed_dev = max(max_speed_dev, maximum(abs, traj_multi[si, :]))
        end
        ok_deviate = max_speed_dev > 1e-6
    end
    push!(criteria, Dict("name" => "speeds_deviate",
                         "value" => max_speed_dev, "threshold" => 1e-6, "passed" => ok_deviate))
    if !ok_deviate; all_passed = false; end

    # Criterion 3: second fault has distinct effect
    ok_distinct = false
    diff_after_fault2 = 0.0
    if ok_run
        ps_single, _, traj_single = run_single_fault()
        speed_idxs_s = GradPower.gen_speeds(ps_single)
        # Compare after fault 2 onset (step 60 at dt=1/120, t=0.5)
        step_after = Int(round(0.5 / (1.0/120.0))) + 1
        for (gi, si) in enumerate(speed_idxs)
            for ti in step_after:min(size(traj_multi, 2), size(traj_single, 2))
                d = abs(traj_multi[si, ti] - traj_single[speed_idxs_s[gi], ti])
                diff_after_fault2 = max(diff_after_fault2, d)
            end
        end
        ok_distinct = diff_after_fault2 > 1e-6
    end
    push!(criteria, Dict("name" => "fault2_distinct",
                         "value" => diff_after_fault2, "threshold" => 1e-6, "passed" => ok_distinct))
    if !ok_distinct; all_passed = false; end

    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 8, "G2", all_passed, criteria, metadata)
    println("Phase 8 G2: passed=$all_passed  -> $OUT")
    for c in criteria
        println("  $(c["name"]): value=$(c["value"]) threshold=$(c["threshold"]) passed=$(c["passed"])")
    end
    return all_passed
end

exit(main() ? 0 : 1)
