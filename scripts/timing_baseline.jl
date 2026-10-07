#!/usr/bin/env julia
# Phase 6 D1/D3 — Timing baselines for 2-bus, IEEE-9, IEEE-39, ACTIVSg200, ACTIVSg2000.
#
# Loads each case, runs integrate!, records wall-clock time, and writes
# artifacts/phase6/baseline.json.
#
# Run from repo root:
#   julia --project=. scripts/timing_baseline.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GradPower
using Statistics

const REPO = abspath(joinpath(@__DIR__, ".."))
const OUT  = joinpath(REPO, "artifacts", "phase6", "baseline.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "verify", "_phase3_common.jl"))

struct TimingCase
    name::String
    raw::String
    dyr::String
    fault_bus_ext::Int      # external PSS/E bus number
    rfault::Float64
    ton::Float64
    toff::Float64
    dt::Float64
    tend::Float64
    set_zipload_alpha::Bool
    nreps::Int
end

const CASES = [
    TimingCase("2bus_genrou",
        joinpath(REPO, "examples", "2bus.raw"),
        joinpath(REPO, "examples", "2bus.dyr"),
        1, 0.02, 0.1, 0.2, 1.0/120.0, 2.0, false, 5),
    TimingCase("ieee9_nogov",
        joinpath(REPO, "examples", "ieee9_v33.raw"),
        joinpath(REPO, "examples", "ieee9bus.dyr"),
        7, 0.02, 0.2, 0.3, 1.0/120.0, 5.0, true, 5),
    TimingCase("ieee39_gov",
        joinpath(REPO, "examples", "IEEE39.raw"),
        joinpath(REPO, "examples", "IEEE39_gov.dyr"),
        30, 0.02, 0.1, 0.2, 1.0/120.0, 2.0, true, 5),
    TimingCase("activs200",
        joinpath(REPO, "examples", "ACTIVSg200.raw"),
        joinpath(REPO, "examples", "ACTIVSg200.dyr"),
        15, 0.02, 0.2, 0.3, 1.0/120.0, 2.0, true, 3),
    TimingCase("activs2000",
        joinpath(REPO, "examples", "ACTIVSg2000.raw"),
        joinpath(REPO, "examples", "ACTIVSg2000.dyr"),
        1001, 0.02, 0.2, 0.3, 1.0/120.0, 2.0, true, 3),
    TimingCase("activs70k",
        joinpath(REPO, "examples", "ACTIVSg70k.raw"),
        joinpath(REPO, "examples", "ACTIVSg70k.dyr"),
        3, 0.02, 0.2, 0.3, 1.0/120.0, 2.0, true, 1),
]

function setup_case(c::TimingCase)
    ps = GradPower.from_psse(c.raw, c.dyr)
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    if c.set_zipload_alpha
        for dev in ps.dynamic.devices
            if dev.dtype isa GradPower.ZIPLoad
                dev.dtype.α = 0.5
            end
        end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    return ps, dp
end

function setup_case_timed(c::TimingCase)
    parse_t0 = time_ns()
    ps = GradPower.from_psse(c.raw, c.dyr)
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    parse_ns = time_ns() - parse_t0
    if c.set_zipload_alpha
        for dev in ps.dynamic.devices
            if dev.dtype isa GradPower.ZIPLoad
                dev.dtype.α = 0.5
            end
        end
    end
    dp = GradPower.DynamicProblem(ps)
    init_t0 = time_ns()
    GradPower.initialize_dynamics!(dp, ps)
    init_ns = time_ns() - init_t0
    return ps, dp, parse_ns, init_ns
end

function _breakdown_dict(log::GradPower.SolverLog)
    Dict{String,Any}(
        "residual"      => Dict{String,Any}("count" => log.residual_count,      "total_s" => log.residual_ns / 1e9),
        "jacobian"      => Dict{String,Any}("count" => log.jacobian_count,      "total_s" => log.jacobian_ns / 1e9),
        "lsolve_factor" => Dict{String,Any}("count" => log.lsolve_factor_count, "total_s" => log.lsolve_factor_ns / 1e9),
        "lsolve_solve"  => Dict{String,Any}("count" => log.lsolve_solve_count,  "total_s" => log.lsolve_solve_ns / 1e9),
        "ybus_mul"      => Dict{String,Any}("count" => log.ybus_mul_count,      "total_s" => log.ybus_mul_ns / 1e9),
        "init"          => Dict{String,Any}("total_s" => log.init_ns / 1e9),
        "parse"         => Dict{String,Any}("total_s" => log.parse_ns / 1e9),
    )
end

function run_timed(c::TimingCase, ps, dp)
    z0 = copy(dp.zvec)
    nbus = length(ps.buses)
    sys_dim = ps.dynamic.diff_dim + ps.dynamic.alg_dim + 2*nbus

    # Map external fault bus to internal index
    fault_bus_int = ps.busmap[c.fault_bus_ext]

    # Warmup run (without log)
    dp.zvec .= z0
    empty!(ps.dynamic.events)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, c.rfault, c.ton, c.toff))
    GradPower.integrate!(dp, ps, c.tend; dt=c.dt)

    # Timed runs with SolverLog
    times = Float64[]
    best_log = nothing
    best_time = Inf
    for _ in 1:c.nreps
        dp.zvec .= z0
        empty!(ps.dynamic.events)
        GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, c.rfault, c.ton, c.toff))
        GC.gc()
        slog = GradPower.SolverLog()
        t0 = time_ns()
        GradPower.integrate!(dp, ps, c.tend; dt=c.dt, log=slog)
        elapsed = (time_ns() - t0) / 1e9
        push!(times, elapsed)
        if elapsed < best_time
            best_time = elapsed
            best_log = slog
        end
    end

    # Setup-timed run (for init/parse timing) — run once since setup is one-shot
    _, _, parse_ns, init_ns = setup_case_timed(c)
    best_log.parse_ns = parse_ns
    best_log.init_ns = init_ns

    return Dict{String,Any}(
        "name"       => c.name,
        "sys_dim"    => sys_dim,
        "nsteps"     => Int(round(c.tend / c.dt)),
        "tend"       => c.tend,
        "dt"         => c.dt,
        "t_min_s"    => minimum(times),
        "t_median_s" => median(times),
        "t_max_s"    => maximum(times),
        "nreps"      => c.nreps,
        "completed"  => true,
        "breakdown"  => _breakdown_dict(best_log),
    )
end

function main()
    results = Dict{String,Any}[]
    all_ok = true

    for c in CASES
        print("  $(c.name) ... ")
        flush(stdout)
        t0 = time()
        try
            ps, dp = setup_case(c)
            r = run_timed(c, ps, dp)
            push!(results, r)
            println("OK  min=$(round(r["t_min_s"]*1000, digits=1))ms  dim=$(r["sys_dim"])")
        catch e
            all_ok = false
            push!(results, Dict{String,Any}(
                "name" => c.name, "completed" => false,
                "error" => sprint(showerror, e)))
            println("FAIL  ", sprint(showerror, e))
        end
    end

    # Write baseline.json
    metadata = Dict{String,Any}(
        "hardware"   => string(Sys.cpu_info()[1].model),
        "git_sha"    => git_sha(REPO),
        "julia"      => string(VERSION),
        "nthreads"   => Threads.nthreads(),
    )
    out = Dict{String,Any}(
        "cases"    => results,
        "metadata" => metadata,
        "all_completed" => all_ok,
    )
    open(OUT, "w") do io
        write(io, pretty(json_value(out)))
        write(io, "\n")
    end
    println("\nBaseline written to $OUT")
    return all_ok
end

main()
