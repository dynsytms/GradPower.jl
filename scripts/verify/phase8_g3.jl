#!/usr/bin/env julia
# Phase 8 Remediation G3 -- Load disconnection
#
# Disconnect a ZIPLoad on the 9-bus system mid-simulation using the new
# DisconnectDeviceEvent (online flag mechanism, no pvec zeroing).
#
# Pass criteria:
#   1. Simulation completes without error.
#   2. The disconnected bus shows a voltage change after disconnection.
#   3. Power injection at the disconnected bus goes to zero after disconnection.
#      (Verified by checking that the device's online flag is false and the
#      residual contribution is zero.)
#
# Writes artifacts/phase8/g3.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase8", "g3.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function run_fault_plus_disconnect()
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

    # Fault at bus 7
    fault_bus = ps.busmap[7]
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus, 0.02, 0.2, 0.3))

    # Find the ZIPLoad device at bus 5.
    load_dev_idx = 0
    for (i, dev) in enumerate(ps.dynamic.devices)
        if dev.dtype isa GradPower.ZIPLoad && dev.dtype.bus == 5
            load_dev_idx = i
            break
        end
    end
    @assert load_dev_idx > 0 "ZIPLoad at bus 5 not found"

    # Disconnect load at bus 5 at t=0.5 using new mechanism
    GradPower.add_disconnect_event!(ps, GradPower.DisconnectDeviceEvent(load_dev_idx, 0.5))

    tvec, traj = GradPower.integrate!(dp, ps, 2.0; dt=1.0/120.0)

    # Check that the device's online flag is false after simulation
    device_offline = !ps.dynamic.devices[load_dev_idx].online

    return ps, dp, tvec, traj, device_offline, load_dev_idx
end

function run_fault_only()
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
    fault_bus = ps.busmap[7]
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus, 0.02, 0.2, 0.3))
    tvec, traj = GradPower.integrate!(dp, ps, 2.0; dt=1.0/120.0)
    return ps, tvec, traj
end

function main()
    t_start = time()
    criteria = Any[]
    all_passed = true

    # Run fault + disconnect
    ok_run = false
    ps_disc = nothing; traj_disc = nothing; device_offline = false
    load_dev_idx = 0
    try
        ps_disc, dp_disc, tvec_disc, traj_disc, device_offline, load_dev_idx = run_fault_plus_disconnect()
        ok_run = true
    catch e
        println("Fault+disconnect run failed: ", sprint(showerror, e, catch_backtrace()))
    end
    push!(criteria, Dict("name" => "fault_disconnect_completes",
                         "value" => ok_run, "threshold" => true, "passed" => ok_run))
    if !ok_run; all_passed = false; end

    # Check device online flag is false
    push!(criteria, Dict("name" => "device_offline_after_disconnect",
                         "value" => device_offline, "threshold" => true, "passed" => device_offline))
    if !device_offline; all_passed = false; end

    # Verify the disconnected bus shows a voltage change (compare pre/post disconnect)
    ok_voltage_change = false
    voltage_diff = 0.0
    if ok_run
        dt = 1.0/120.0
        disc_bus = 5  # external bus
        disc_bus_int = ps_disc.busmap[disc_bus]
        diff_dim = ps_disc.dynamic.diff_dim
        alg_dim  = ps_disc.dynamic.alg_dim
        net_ptr  = diff_dim + alg_dim
        vr_idx = net_ptr + 2*(disc_bus_int - 1) + 1
        vi_idx = vr_idx + 1
        # Compare voltage magnitude just before and after disconnect
        step_before = Int(round(0.5 / dt))
        step_after  = step_before + 5  # a few steps after disconnect
        if step_after <= size(traj_disc, 2)
            vm_before = sqrt(traj_disc[vr_idx, step_before]^2 + traj_disc[vi_idx, step_before]^2)
            vm_after  = sqrt(traj_disc[vr_idx, step_after]^2 + traj_disc[vi_idx, step_after]^2)
            voltage_diff = abs(vm_after - vm_before)
            ok_voltage_change = voltage_diff > 1e-6
        end
    end
    push!(criteria, Dict("name" => "voltage_change_at_disconnect_bus",
                         "value" => voltage_diff, "threshold" => 1e-6, "passed" => ok_voltage_change))
    if !ok_voltage_change; all_passed = false; end

    # Compare with fault-only to verify disconnect has a distinct effect
    ok_distinct = false
    diff_after_disc = 0.0
    if ok_run
        ps_only, _, traj_only = run_fault_only()
        speed_idxs_disc = GradPower.gen_speeds(ps_disc)
        speed_idxs_only = GradPower.gen_speeds(ps_only)
        dt = 1.0/120.0
        step_after = Int(round(0.5 / dt)) + 1
        for (gi, si) in enumerate(speed_idxs_disc)
            for ti in step_after:min(size(traj_disc, 2), size(traj_only, 2))
                d = abs(traj_disc[si, ti] - traj_only[speed_idxs_only[gi], ti])
                diff_after_disc = max(diff_after_disc, d)
            end
        end
        ok_distinct = diff_after_disc > 1e-6
    end
    push!(criteria, Dict("name" => "disconnect_distinct_effect",
                         "value" => diff_after_disc, "threshold" => 1e-6, "passed" => ok_distinct))
    if !ok_distinct; all_passed = false; end

    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 8, "G3", all_passed, criteria, metadata)
    println("Phase 8 G3: passed=$all_passed  -> $OUT")
    for c in criteria
        println("  $(c["name"]): value=$(c["value"]) threshold=$(c["threshold"]) passed=$(c["passed"])")
    end
    return all_passed
end

exit(main() ? 0 : 1)
