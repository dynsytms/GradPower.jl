#!/usr/bin/env julia
#
# Phase 6: Comprehensive benchmark — CPU batched vs GPU shared-factor
# Across system sizes (200, 2000, 70k) and batch sizes (M=1..256+).

using GradPower, CUDA, CUDSS, Printf
const ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

function bench_system(case_raw, case_dyr, label; M_values=[1, 4, 16, 64, 128, 256],
                      tf=1.0, dt=1.0/120.0, tol=1e-10)
    println("=" ^ 70)
    println("  $label")
    println("=" ^ 70)
    flush(stdout)

    ps = GradPower.from_psse(case_raw, case_dyr)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    nbus = length(ps.buses)
    sys_dim = length(dp.zvec)
    @printf("  %d buses, %d unknowns\n\n", nbus, sys_dim)
    flush(stdout)

    @printf("  %-6s  %10s  %10s  %10s  %10s  %10s\n",
            "M", "CPU(s)", "GPU-SF(s)", "Speedup", "CPU sc/s", "GPU sc/s")
    @printf("  %s\n", "-"^66)
    flush(stdout)

    results = []

    for M in M_values
        fault_buses = [1 + (m - 1) % nbus for m in 1:M]
        rfaults = fill(0.02, M)

        # CPU batched
        empty!(ps.dynamic.events)
        GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))
        bl = GradPower.BatchedLayout(dp, ps, M)
        GradPower.integrate_batched!(bl, ps, 0.05; dt=dt, newton_tol=tol)
        bl2 = GradPower.BatchedLayout(dp, ps, M)
        t0 = time_ns()
        GradPower.integrate_batched!(bl2, ps, tf; dt=dt, newton_tol=tol)
        t_cpu = (time_ns() - t0) / 1e9

        # GPU shared-factor with per-scenario faults
        empty!(ps.dynamic.events)
        local t_gpu
        try
            gbl = ext.GpuBatchedLayout(dp, ps, M)
            ext.integrate_gpu_shared_multi!(gbl, ps, 0.05, fault_buses, rfaults,
                                             0.1, 0.2; dt=dt, newton_tol=tol)
            CUDA.synchronize()
            gbl2 = ext.GpuBatchedLayout(dp, ps, M)
            CUDA.synchronize()
            t0 = time_ns()
            ext.integrate_gpu_shared_multi!(gbl2, ps, tf, fault_buses, rfaults,
                                             0.1, 0.2; dt=dt, newton_tol=tol)
            CUDA.synchronize()
            t_gpu = (time_ns() - t0) / 1e9
        catch e
            @printf("  %-6d  %10.1f  %10s  %10s  %10.1f  %10s\n",
                    M, t_cpu, "OOM", "-", M/t_cpu, "-")
            flush(stdout)
            push!(results, (M=M, t_cpu=t_cpu, t_gpu=NaN, speedup=NaN))
            continue
        end

        speedup = t_cpu / t_gpu
        push!(results, (M=M, t_cpu=t_cpu, t_gpu=t_gpu, speedup=speedup))
        @printf("  %-6d  %10.1f  %10.1f  %10.1fx  %10.1f  %10.1f\n",
                M, t_cpu, t_gpu, speedup, M/t_cpu, M/t_gpu)
        flush(stdout)
    end

    println()
    flush(stdout)
    return results
end

# ═══════════════════════════════════════════════════════════════════════

println()
println("╔══════════════════════════════════════════════════════════════════════╗")
println("║  Phase 6: Full Benchmark — CPU Batched vs GPU Shared-Factor        ║")
println("║  Per-scenario faults, Quadro GV100 (32 GB)                         ║")
println("╚══════════════════════════════════════════════════════════════════════╝")
println()
flush(stdout)

r200 = bench_system("examples/ACTIVSg200.raw", "examples/ACTIVSg200.dyr",
                     "ACTIVSg200",
                     M_values=[1, 4, 16, 64, 128, 256])

r2k = bench_system("examples/ACTIVSg2000.raw", "examples/ACTIVSg2000.dyr",
                    "ACTIVSg2000",
                    M_values=[1, 4, 16, 64, 128, 256])

r70k = bench_system("examples/ACTIVSg70k.raw", "examples/ACTIVSg70k.dyr",
                     "ACTIVSg70k",
                     M_values=[1, 4, 16, 64])

# ── Summary ──
println("=" ^ 70)
println("  SUMMARY: Best throughput per system")
println("=" ^ 70)
println()
for (label, results) in [("ACTIVSg200", r200), ("ACTIVSg2000", r2k), ("ACTIVSg70k", r70k)]
    valid = filter(r -> !isnan(r.t_gpu), results)
    if isempty(valid)
        println("  $label: no valid GPU results")
        continue
    end
    best = argmax(r -> r.M / r.t_gpu, valid)
    best_r = valid[best]
    cpu_at_same_M = best_r.t_cpu
    @printf("  %-14s  M=%-4d  GPU: %.1f scen/s  CPU: %.1f scen/s  Speedup: %.1fx\n",
            label, best_r.M, best_r.M / best_r.t_gpu,
            best_r.M / cpu_at_same_M, best_r.speedup)
end
println()

# Polaris projection
println("  Polaris projection (4× A100 per node, ~1.5× GV100 perf):")
for (label, results) in [("ACTIVSg2000", r2k), ("ACTIVSg70k", r70k)]
    valid = filter(r -> !isnan(r.t_gpu), results)
    isempty(valid) && continue
    best = argmax(r -> r.M / r.t_gpu, valid)
    best_r = valid[best]
    gv100_scps = best_r.M / best_r.t_gpu
    a100_scps = gv100_scps * 1.5
    node_scps = a100_scps * 4
    @printf("    %-14s  %.0f scen/s per A100  →  %.0f scen/s per node  →  %dk scen/hr\n",
            label, a100_scps, node_scps, Int(round(node_scps * 3600 / 1000)))
end
println()
flush(stdout)
