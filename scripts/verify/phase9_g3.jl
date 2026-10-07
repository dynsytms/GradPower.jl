#!/usr/bin/env julia
# Phase 9 G3 -- Trajectory match against ANDES reference
#
# 2-bus case with Genrou + SEXS + IEEEST (mode 1), fault at bus 1 (rf=1.0,
# ton=0.1, toff=0.2). Compares generator speed deviation (w) trajectory
# against ANDES reference saved in artifacts/phase9/ref_ieeest.npz.
#
# Pass: max |trajectory - reference|_inf / |reference|_inf <= 1e-3.
#
# Writes artifacts/phase9/g3.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using NPZ

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase9", "g3.json")
const REF  = joinpath(REPO, "artifacts", "phase9", "ref_ieeest.npz")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function interp1d(t_src, y_src, t_dst)
    n = length(t_dst)
    y_dst = zeros(n)
    for i in 1:n
        t = t_dst[i]
        if t <= t_src[1]
            y_dst[i] = y_src[1]
        elseif t >= t_src[end]
            y_dst[i] = y_src[end]
        else
            # binary search
            lo, hi = 1, length(t_src)
            while hi - lo > 1
                mid = (lo + hi) >> 1
                if t_src[mid] <= t
                    lo = mid
                else
                    hi = mid
                end
            end
            frac = (t - t_src[lo]) / (t_src[hi] - t_src[lo])
            y_dst[i] = y_src[lo] + frac * (y_src[hi] - y_src[lo])
        end
    end
    return y_dst
end

function main()
    t_start = time()
    criteria = Any[]
    all_passed = true
    tol = 1e-3

    if !isfile(REF)
        println("ERROR: Reference file not found: $REF")
        println("Generate it with: cd uqgrid && python3 gen_ieeest_ref.py")
        exit(1)
    end

    ref = npzread(REF)
    t_ref = ref["tvec"]
    w_ref = ref["w"]

    # Run Julia simulation
    ps = GradPower.from_psse(
        joinpath(REPO, "examples", "2bus.raw"),
        joinpath(REPO, "examples", "2bus_IEEEST.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    fault_bus_int = ps.busmap[1]
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, 1.0, 0.1, 0.2))
    tvec, traj = GradPower.integrate!(dp, ps, 2.0; dt=1.0/120.0)

    # Extract generator w (speed deviation) — Genrou diff state 5
    w_julia = traj[5, :]

    # Interpolate ANDES reference onto Julia time grid
    w_ref_interp = interp1d(t_ref, w_ref, tvec)

    # Compute relative error: max|w_julia - w_ref| / max(|w_ref|, 1e-10)
    max_diff = maximum(abs, w_julia .- w_ref_interp)
    ref_norm = maximum(abs, w_ref)
    rel_err = max_diff / max(ref_norm, 1e-10)

    ok = rel_err <= tol
    push!(criteria, Dict("name" => "w_traj_rel_err",
                         "value" => rel_err, "threshold" => tol, "passed" => ok))
    if !ok; all_passed = false; end
    println("  w_traj: max_diff=$max_diff, ref_norm=$ref_norm, rel_err=$rel_err ", ok ? "PASS" : "FAIL")

    # Also check delta trajectory if available
    if haskey(ref, "delta")
        delta_ref = ref["delta"]
        delta_julia = traj[6, :]
        delta_ref_interp = interp1d(t_ref, delta_ref, tvec)
        max_diff_d = maximum(abs, delta_julia .- delta_ref_interp)
        ref_norm_d = maximum(abs, delta_ref)
        rel_err_d = max_diff_d / max(ref_norm_d, 1e-10)

        ok_d = rel_err_d <= tol
        push!(criteria, Dict("name" => "delta_traj_rel_err",
                             "value" => rel_err_d, "threshold" => tol, "passed" => ok_d))
        if !ok_d; all_passed = false; end
        println("  delta_traj: max_diff=$max_diff_d, ref_norm=$ref_norm_d, rel_err=$rel_err_d ", ok_d ? "PASS" : "FAIL")
    end

    # Check vsout trajectory if available
    if haskey(ref, "vsout")
        vs_ref = ref["vsout"]
        diff_dim = ps.dynamic.diff_dim
        vs_z = diff_dim + Int(ps.dynamic.layout.ieeest.alg_ptr[1])
        vs_julia = traj[vs_z, :]
        vs_ref_interp = interp1d(t_ref, vs_ref, tvec)
        max_diff_v = maximum(abs, vs_julia .- vs_ref_interp)
        ref_norm_v = maximum(abs, vs_ref)
        rel_err_v = max_diff_v / max(ref_norm_v, 1e-10)

        ok_v = rel_err_v <= tol
        push!(criteria, Dict("name" => "vsout_traj_rel_err",
                             "value" => rel_err_v, "threshold" => tol, "passed" => ok_v))
        if !ok_v; all_passed = false; end
        println("  vsout_traj: max_diff=$max_diff_v, ref_norm=$ref_norm_v, rel_err=$rel_err_v ", ok_v ? "PASS" : "FAIL")
    end

    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 9, "G3", all_passed, criteria, metadata)
    println("\nPhase 9 G3: passed=$all_passed  -> $OUT")
    return all_passed
end

exit(main() ? 0 : 1)
