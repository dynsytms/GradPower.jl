#!/usr/bin/env julia
# Phase 8 Remediation G4 -- Line trip
#
# Trip a line on the 9-bus system mid-simulation using TripLineEvent.
# Verifies Ybus is modified and post-trip voltages differ from pre-trip.
#
# Pass criteria:
#   1. Simulation completes without error.
#   2. Ybus is modified (the tripped line's entries are removed).
#   3. Post-trip voltage magnitudes differ from pre-trip.
#
# Writes artifacts/phase8/g4.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using SparseArrays

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase8", "g4.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function run_trip_line()
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

    # Save Ybus before trip for comparison
    ybus_before = copy(ps.network.ybus_real.nzval)

    # Trip line 7-8 at t=0.5
    trip_event = GradPower.create_trip_line_event(ps, 7, 8, 0.5)
    GradPower.add_trip_event!(ps, trip_event)

    tvec, traj = GradPower.integrate!(dp, ps, 2.0; dt=1.0/120.0)

    ybus_after = copy(ps.network.ybus_real.nzval)

    return ps, dp, tvec, traj, ybus_before, ybus_after
end

function run_no_trip()
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
    tvec, traj = GradPower.integrate!(dp, ps, 2.0; dt=1.0/120.0)
    return ps, tvec, traj
end

function main()
    t_start = time()
    criteria = Any[]
    all_passed = true

    # Criterion 1: simulation completes
    ok_run = false
    ps_trip = nothing; traj_trip = nothing
    ybus_before = Float64[]; ybus_after = Float64[]
    try
        ps_trip, dp_trip, tvec_trip, traj_trip, ybus_before, ybus_after = run_trip_line()
        ok_run = true
    catch e
        println("Line trip run failed: ", sprint(showerror, e, catch_backtrace()))
    end
    push!(criteria, Dict("name" => "line_trip_completes",
                         "value" => ok_run, "threshold" => true, "passed" => ok_run))
    if !ok_run; all_passed = false; end

    # Criterion 2: Ybus is modified
    ok_ybus = false
    ybus_diff = 0.0
    if ok_run
        ybus_diff = maximum(abs, ybus_after .- ybus_before)
        ok_ybus = ybus_diff > 1e-10
    end
    push!(criteria, Dict("name" => "ybus_modified",
                         "value" => ybus_diff, "threshold" => 1e-10, "passed" => ok_ybus))
    if !ok_ybus; all_passed = false; end

    # Criterion 3: post-trip voltage magnitudes differ from no-trip case
    ok_voltage = false
    max_vm_diff = 0.0
    if ok_run
        ps_notrip, _, traj_notrip = run_no_trip()
        diff_dim = ps_trip.dynamic.diff_dim
        alg_dim = ps_trip.dynamic.alg_dim
        net_ptr = diff_dim + alg_dim
        nbus = length(ps_trip.buses)
        dt = 1.0/120.0
        # Compare voltage magnitudes at the end of simulation
        for bus in 1:nbus
            vr_idx = net_ptr + 2*(bus - 1) + 1
            vi_idx = vr_idx + 1
            vm_trip   = sqrt(traj_trip[vr_idx, end]^2 + traj_trip[vi_idx, end]^2)
            vm_notrip = sqrt(traj_notrip[vr_idx, end]^2 + traj_notrip[vi_idx, end]^2)
            max_vm_diff = max(max_vm_diff, abs(vm_trip - vm_notrip))
        end
        ok_voltage = max_vm_diff > 1e-6
    end
    push!(criteria, Dict("name" => "post_trip_voltage_differs",
                         "value" => max_vm_diff, "threshold" => 1e-6, "passed" => ok_voltage))
    if !ok_voltage; all_passed = false; end

    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 8, "G4", all_passed, criteria, metadata)
    println("Phase 8 G4: passed=$all_passed  -> $OUT")
    for c in criteria
        println("  $(c["name"]): value=$(c["value"]) threshold=$(c["threshold"]) passed=$(c["passed"])")
    end
    return all_passed
end

exit(main() ? 0 : 1)
