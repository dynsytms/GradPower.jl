#!/usr/bin/env julia
# Phase 12 G3 — GMRES iteration count diagnostic.
#
# Records median/max GMRES iterations per Newton step for ACTIVSg200,
# ACTIVSg2000, and ACTIVSg70k. Reports separately for fault-on and
# fault-off periods. Flags if median > 15 outside fault periods or
# max > 50 during fault-on. No hard pass/fail — this is diagnostic data.
#
# Writes artifacts/phase12/g3.json. Always exits 0 (diagnostic gate).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using Statistics

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase12", "g3.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

"""
Run a fault case and return (slog, fault_on_iters, fault_off_iters).

The GMRES iters are split into fault-on (steps between ton and toff)
and fault-off (all other steps). Since the SolverLog records one entry
per Newton iteration (not per time step), we approximate by classifying
based on Newton iteration index ranges.
"""
function run_case_diagnostic(raw, dyr, fault_bus; tend=1.0, ton=0.1, toff=0.2)
    ps = from_psse(joinpath(REPO, "examples", raw), joinpath(REPO, "examples", dyr))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus, 0.02, ton, toff))
    slog = GradPower.SolverLog()
    tvec, traj = GradPower.integrate!(dp, ps, tend; dt=1.0/120.0,
                                       solver=:schur_gmres, log=slog,
                                       newton_tol=1e-10)
    for ev in ps.dynamic.events; GradPower.deactivate!(ev); end

    # Classify GMRES iters into fault-on/off periods.
    # Each time step produces 2-4 Newton iterations (normal + event re-solve).
    # We use the time step index to classify.
    dt = 1.0/120.0
    step_on  = Int(round(ton / dt))
    step_off = Int(round(toff / dt))
    nsteps   = Int(round(tend / dt))

    # Count Newton iters per step by running the same logic as integrate!
    # Approximate: each normal step has ~2 Newton iters, event steps have ~4.
    # Instead, just split the gmres_iters array proportionally based on step range.
    # Better approach: use step counts to estimate Newton iter boundaries.
    total_iters = length(slog.gmres_iters)
    iters_per_step = total_iters / nsteps

    # Fault-on steps: step_on to step_off (plus event re-solves)
    # Rough split: fault-on spans step_on..step_off out of 1..nsteps
    on_start  = max(1, Int(round((step_on / nsteps) * total_iters)))
    on_end    = min(total_iters, Int(round((step_off / nsteps) * total_iters)))

    fault_on_iters  = slog.gmres_iters[on_start:on_end]
    fault_off_iters = vcat(slog.gmres_iters[1:max(1,on_start-1)],
                           slog.gmres_iters[min(total_iters,on_end+1):end])

    return slog, fault_on_iters, fault_off_iters
end

function main()
    t0 = time()
    criteria = Any[]

    cases = [
        ("activs200",  "ACTIVSg200.raw",  "ACTIVSg200.dyr",  1),
        ("activs2000", "ACTIVSg2000.raw", "ACTIVSg2000.dyr", 1),
        ("activs70k",  "ACTIVSg70k.raw",  "ACTIVSg70k.dyr",  1),
    ]

    for (name, raw, dyr, fbus) in cases
        println("  $name: running Schur-GMRES diagnostic...")
        slog, on_iters, off_iters = run_case_diagnostic(raw, dyr, fbus)

        all_iters = slog.gmres_iters
        med_all  = isempty(all_iters)  ? 0.0 : median(all_iters)
        max_all  = isempty(all_iters)  ? 0   : maximum(all_iters)
        med_on   = isempty(on_iters)   ? 0.0 : median(on_iters)
        max_on   = isempty(on_iters)   ? 0   : maximum(on_iters)
        med_off  = isempty(off_iters)  ? 0.0 : median(off_iters)
        max_off  = isempty(off_iters)  ? 0   : maximum(off_iters)

        println("  $name: all iters=$(length(all_iters)), median=$med_all, max=$max_all")
        println("  $name: fault-on  median=$med_on, max=$max_on")
        println("  $name: fault-off median=$med_off, max=$max_off")

        # Diagnostic flags (not hard pass/fail)
        off_flag = med_off > 15
        on_flag  = max_on > 50

        push!(criteria, Dict("name" => "$(name)_median_all",     "value" => med_all,  "threshold" => nothing, "passed" => true))
        push!(criteria, Dict("name" => "$(name)_max_all",        "value" => max_all,  "threshold" => nothing, "passed" => true))
        push!(criteria, Dict("name" => "$(name)_median_faulton", "value" => med_on,   "threshold" => nothing, "passed" => true))
        push!(criteria, Dict("name" => "$(name)_max_faulton",    "value" => max_on,   "threshold" => 50,      "passed" => !on_flag))
        push!(criteria, Dict("name" => "$(name)_median_faultoff","value" => med_off,  "threshold" => 15,      "passed" => !off_flag))

        if off_flag
            println("  WARNING: $name fault-off median GMRES iters ($med_off) > 15")
        end
        if on_flag
            println("  WARNING: $name fault-on max GMRES iters ($max_on) > 50")
        end
    end

    metadata = Dict{String,Any}(
        "hardware"    => string(Sys.cpu_info()[1].model),
        "git_sha"     => git_sha(REPO),
        "wallclock_s" => time() - t0,
    )
    # Diagnostic gate: always passes (data collection only)
    write_artifact(OUT, 12, "G3", true, criteria, metadata)
    println("Phase 12 G3: passed=true (diagnostic)  -> $OUT")
    return true
end

exit(main() ? 0 : 1)
