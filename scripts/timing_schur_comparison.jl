#!/usr/bin/env julia
# Timing comparison: monolithic KLU vs Schur-reduced solver.
# Backs up the previous baseline as baseline_phase10.json (if not already done),
# then writes artifacts/phase6/baseline.json with both solver timings.
#
# Run from repo root:
#   julia --project=. scripts/timing_schur_comparison.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GradPower
using Statistics

const REPO = abspath(joinpath(@__DIR__, ".."))
const OUT  = joinpath(REPO, "artifacts", "phase11", "timing_comparison.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "verify", "_phase3_common.jl"))

struct TimingCase
    name::String
    raw::String
    dyr::String
    fault_bus_ext::Int
    rfault::Float64
    ton::Float64
    toff::Float64
    dt::Float64
    tend::Float64
    set_zipload_alpha::Bool
    nreps::Int
end

const CASES = [
    TimingCase("ieee9_nogov",
        joinpath(REPO, "examples", "ieee9_v33.raw"),
        joinpath(REPO, "examples", "ieee9bus.dyr"),
        7, 0.02, 0.2, 0.3, 1.0/120.0, 5.0, true, 5),
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

function run_solver(c::TimingCase, ps, dp, z0, solver::Symbol)
    nbus = length(ps.buses)
    sys_dim = ps.dynamic.diff_dim + ps.dynamic.alg_dim + 2*nbus
    fault_bus_int = ps.busmap[c.fault_bus_ext]

    # Warmup
    dp.zvec .= z0
    empty!(ps.dynamic.events)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, c.rfault, c.ton, c.toff))
    GradPower.integrate!(dp, ps, c.tend; dt=c.dt, solver=solver)

    # Timed runs
    times = Float64[]
    for _ in 1:c.nreps
        dp.zvec .= z0
        empty!(ps.dynamic.events)
        GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, c.rfault, c.ton, c.toff))
        GC.gc()
        t0 = time_ns()
        GradPower.integrate!(dp, ps, c.tend; dt=c.dt, solver=solver)
        elapsed = (time_ns() - t0) / 1e9
        push!(times, elapsed)
    end

    return Dict{String,Any}(
        "solver"     => String(solver),
        "sys_dim"    => sys_dim,
        "nsteps"     => Int(round(c.tend / c.dt)),
        "t_min_s"    => minimum(times),
        "t_median_s" => median(times),
        "t_max_s"    => maximum(times),
        "nreps"      => c.nreps,
    )
end

function main()
    results = Dict{String,Any}[]

    for c in CASES
        println("$(c.name):")
        flush(stdout)
        try
            ps, dp = setup_case(c)
            z0 = copy(dp.zvec)

            mono = run_solver(c, ps, dp, z0, :monolithic)
            schur = run_solver(c, ps, dp, z0, :schur)

            ratio = schur["t_min_s"] / mono["t_min_s"]
            println("  monolithic: $(round(mono["t_min_s"]*1000, digits=1)) ms")
            println("  schur:      $(round(schur["t_min_s"]*1000, digits=1)) ms")
            println("  ratio:      $(round(ratio, digits=2))x")

            push!(results, Dict{String,Any}(
                "name"       => c.name,
                "monolithic" => mono,
                "schur"      => schur,
                "ratio_schur_over_mono" => ratio,
            ))
        catch e
            println("  FAIL: ", sprint(showerror, e))
            push!(results, Dict{String,Any}(
                "name" => c.name, "error" => sprint(showerror, e)))
        end
    end

    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "julia"    => string(VERSION),
        "nthreads" => Threads.nthreads(),
    )
    out = Dict{String,Any}(
        "cases"    => results,
        "metadata" => metadata,
    )
    open(OUT, "w") do io
        write(io, pretty(json_value(out)))
        write(io, "\n")
    end
    println("\nComparison written to $OUT")
end

main()
