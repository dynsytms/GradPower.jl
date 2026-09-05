#!/usr/bin/env julia
# Phase 8 Remediation G1 -- Single-fault bit-identical (regression gate)
#
# All existing single-fault reference cases produce identical trajectories
# after the device disconnection remediation. Runs the 2-bus and IEEE-9
# cases and verifies they still match the uqgrid reference.
#
# Writes artifacts/phase8/g1.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using NPZ

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase8", "g1.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function frozen_ref(name)
    p = joinpath(REPO, "artifacts", "phase0", "references", name)
    isfile(p) || (p = joinpath(REPO, "examples", "refs", name))
    return p
end

# --- Case 1: 2bus GENROU ---
function run_2bus()
    ps = GradPower.from_psse(
        joinpath(REPO, "examples", "2bus.raw"),
        joinpath(REPO, "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))
    tvec, traj = GradPower.integrate!(dp, ps, 2.0; dt=1.0/120.0)

    ref = npzread(frozen_ref("2bus_genrou.npz"))
    hist = ref["history"]
    compare_rows = [
        (1,  1), (2,  2), (3,  3), (4,  4), (5,  5), (6,  6),
        (7,  9), (8, 10), (9, 11), (10, 12),
        (11, 15), (12, 16), (13, 17), (14, 18),
    ]
    worst = 0.0
    for (jr, pr) in compare_rows
        d = maximum(abs, traj[jr, :] .- hist[pr, :])
        worst = max(worst, d)
    end
    return worst
end

# --- Case 2: IEEE 9-bus ---
function run_ieee9()
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
    fault_bus_int = ps.busmap[7]
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, 0.02, 0.2, 0.3))
    tvec, traj = GradPower.integrate!(dp, ps, 5.0; dt=1.0/120.0)

    ref = npzread(frozen_ref("ieee9_nogov.npz"))
    hist = ref["history"]
    speed_idx_py = Int.(ref["speed_idx"])
    julia_speeds = GradPower.gen_speeds(ps)

    worst = 0.0
    for (gi, jsi) in enumerate(julia_speeds)
        py_row = speed_idx_py[gi] + 1
        for ti in 1:min(size(traj, 2), size(hist, 2))
            d = abs(traj[jsi, ti] - hist[py_row, ti])
            worst = max(worst, d)
        end
    end
    return worst
end

function main()
    t_start = time()
    criteria = Any[]
    all_passed = true

    for (name, runner, tol) in [("2bus_genrou", run_2bus, 5.0e-9),
                                 ("ieee9_nogov", run_ieee9, 1.0e-3)]
        print("  $name ... "); flush(stdout)
        ok = false; err = Inf
        try
            err = runner()
            ok = err < tol
        catch e
            println("ERROR: ", sprint(showerror, e, catch_backtrace()))
        end
        push!(criteria, Dict("name" => "$(name)_identical",
                             "value" => err, "threshold" => tol, "passed" => ok))
        if !ok; all_passed = false; end
        println(ok ? "PASS" : "FAIL", "  max_err=", err)
    end

    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 8, "G1", all_passed, criteria, metadata)
    println("\nPhase 8 G1: passed=$all_passed  -> $OUT")
    return all_passed
end

exit(main() ? 0 : 1)
