#!/usr/bin/env julia
# Instrumented timing comparison: monolithic KLU vs Schur-direct vs Schur-GMRES.
# Records per-component breakdown via SolverLog + GMRES iteration stats.
#
# Run from repo root:
#   julia --project=. scripts/timing_all_solvers.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GradPower
using Statistics
using Printf

const REPO = abspath(joinpath(@__DIR__, ".."))
const OUT  = joinpath(REPO, "artifacts", "phase12", "timing_comparison.json")
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

const SOLVERS = [:monolithic, :schur, :schur_gmres]

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

function _breakdown_dict(log::GradPower.SolverLog)
    d = Dict{String,Any}(
        "residual"      => Dict{String,Any}("count" => log.residual_count,      "total_s" => log.residual_ns / 1e9),
        "jacobian"      => Dict{String,Any}("count" => log.jacobian_count,      "total_s" => log.jacobian_ns / 1e9),
        "lsolve_factor" => Dict{String,Any}("count" => log.lsolve_factor_count, "total_s" => log.lsolve_factor_ns / 1e9),
        "lsolve_solve"  => Dict{String,Any}("count" => log.lsolve_solve_count,  "total_s" => log.lsolve_solve_ns / 1e9),
        "ybus_mul"      => Dict{String,Any}("count" => log.ybus_mul_count,      "total_s" => log.ybus_mul_ns / 1e9),
    )
    if !isempty(log.gmres_iters)
        d["gmres"] = Dict{String,Any}(
            "total_newton_steps" => length(log.gmres_iters),
            "iters_median"      => Int(round(median(log.gmres_iters))),
            "iters_max"         => maximum(log.gmres_iters),
            "iters_min"         => minimum(log.gmres_iters),
            "iters_total"       => sum(log.gmres_iters),
        )
    end
    return d
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
    best_log = nothing
    best_time = Inf
    for _ in 1:c.nreps
        dp.zvec .= z0
        empty!(ps.dynamic.events)
        GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, c.rfault, c.ton, c.toff))
        GC.gc()
        slog = GradPower.SolverLog()
        t0 = time_ns()
        GradPower.integrate!(dp, ps, c.tend; dt=c.dt, solver=solver, log=slog)
        elapsed = (time_ns() - t0) / 1e9
        push!(times, elapsed)
        if elapsed < best_time
            best_time = elapsed
            best_log = slog
        end
    end

    return Dict{String,Any}(
        "solver"     => String(solver),
        "sys_dim"    => sys_dim,
        "nsteps"     => Int(round(c.tend / c.dt)),
        "t_min_s"    => minimum(times),
        "t_median_s" => median(times),
        "t_max_s"    => maximum(times),
        "nreps"      => c.nreps,
        "breakdown"  => _breakdown_dict(best_log),
    )
end

function print_table(case_name::String, results::Dict{String,Any})
    mono  = results["monolithic"]
    schur = results["schur"]
    gmres = results["schur_gmres"]

    mono_t  = mono["t_min_s"] * 1000
    schur_t = schur["t_min_s"] * 1000
    gmres_t = gmres["t_min_s"] * 1000

    println("\n═══ $case_name (dim=$(mono["sys_dim"]), steps=$(mono["nsteps"])) ═══")
    println("┌─────────────────┬────────────┬────────────┬────────────┐")
    println("│                 │ monolithic │ schur-KLU  │ schur-GMRES│")
    println("├─────────────────┼────────────┼────────────┼────────────┤")

    function fmt(v)
        if v < 1.0;       return @sprintf("%8.2f ms", v)
        elseif v < 1000.0; return @sprintf("%8.1f ms", v)
        else;              return @sprintf("%7.2f  s", v/1000)
        end
    end

    println("│ total wall      │ $(fmt(mono_t)) │ $(fmt(schur_t)) │ $(fmt(gmres_t)) │")

    # Breakdown rows
    for (label, key) in [("  residual", "residual"), ("  jacobian", "jacobian"),
                          ("  factor(S/J)", "lsolve_factor"), ("  solve(S/J)", "lsolve_solve"),
                          ("  ybus mul", "ybus_mul")]
        m_s = get(mono["breakdown"], key, nothing)
        s_s = get(schur["breakdown"], key, nothing)
        g_s = get(gmres["breakdown"], key, nothing)
        m_v = m_s !== nothing ? m_s["total_s"] * 1000 : 0.0
        s_v = s_s !== nothing ? s_s["total_s"] * 1000 : 0.0
        g_v = g_s !== nothing ? g_s["total_s"] * 1000 : 0.0
        m_c = m_s !== nothing ? m_s["count"] : 0
        s_c = s_s !== nothing ? s_s["count"] : 0
        g_c = g_s !== nothing ? g_s["count"] : 0

        if m_c == 0 && s_c == 0 && g_c == 0; continue; end
        padlabel = rpad(label, 17)
        println("│ $padlabel │ $(fmt(m_v)) │ $(fmt(s_v)) │ $(fmt(g_v)) │")
    end

    println("├─────────────────┼────────────┼────────────┼────────────┤")
    println("│ ratio vs mono   │     1.00x  │  $(@sprintf("%5.2f", schur_t/mono_t))x  │  $(@sprintf("%5.2f", gmres_t/mono_t))x  │")

    # GMRES iteration stats
    gm = get(gmres["breakdown"], "gmres", nothing)
    if gm !== nothing
        println("├─────────────────┼────────────┼────────────┼────────────┤")
        println("│ GMRES iters     │     —      │     —      │ med=$(gm["iters_median"]) max=$(gm["iters_max"]) │")
    end
    println("└─────────────────┴────────────┴────────────┴────────────┘")
end

function main()
    # Backup existing artifact
    if isfile(OUT)
        backup = replace(OUT, ".json" => "_backup.json")
        cp(OUT, backup; force=true)
        println("Backed up existing artifact to $backup")
    end

    all_results = Dict{String,Any}[]

    for c in CASES
        println("\n--- $(c.name) ---")
        flush(stdout)
        try
            ps, dp = setup_case(c)
            z0 = copy(dp.zvec)

            solver_results = Dict{String,Any}()
            for s in SOLVERS
                print("  $(String(s))... ")
                flush(stdout)
                r = run_solver(c, ps, dp, z0, s)
                solver_results[String(s)] = r
                println("$(round(r["t_min_s"]*1000, digits=1)) ms")
            end

            mono_t = solver_results["monolithic"]["t_min_s"]
            ratios = Dict{String,Any}()
            for s in SOLVERS
                ratios[String(s)] = solver_results[String(s)]["t_min_s"] / mono_t
            end

            case_result = Dict{String,Any}(
                "name"    => c.name,
                "solvers" => solver_results,
                "ratio_vs_monolithic" => ratios,
            )
            push!(all_results, case_result)
            print_table(c.name, solver_results)
        catch e
            println("  FAIL: ", sprint(showerror, e))
            push!(all_results, Dict{String,Any}(
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
        "cases"    => all_results,
        "metadata" => metadata,
    )
    open(OUT, "w") do io
        write(io, pretty(json_value(out)))
        write(io, "\n")
    end
    println("\nComparison written to $OUT")
end

main()
