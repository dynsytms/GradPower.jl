#!/usr/bin/env julia
# Phase 3 G4 — Trajectory parity vs uqgrid for 2bus GENROU+ESDC1A.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using NPZ
using Printf

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const RAW  = joinpath(REPO, "examples", "2bus.raw")
const DYR  = joinpath(REPO, "examples", "2bus_ESDC1A.dyr")
const REF  = joinpath(REPO, "examples", "refs", "2bus_esdc1a.npz")
const OUT  = joinpath(REPO, "artifacts", "phase3", "g4.json")
# Same tolerance shape as test/regression/compare_2bus_genrou.jl (the
# baseline 2bus tolerance frozen at Phase 0).
const TOL  = 5.0e-9
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function main()
    t0 = time()
    ps = from_psse(RAW, DYR)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for dev in ps.dynamic.devices
        if dev.dtype isa GradPower.ZIPLoad; dev.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.2, 0.3))
    tvec, traj = GradPower.integrate!(dp, ps, 5.0; dt=1.0/120.0, verbose=false)

    ref = npzread(REF); hist = ref["history"]
    @assert size(traj, 2) == size(hist, 2) "step count mismatch jl=$(size(traj,2)) py=$(size(hist,2))"

    g = ps.dynamic.layout.genrou
    exc = ps.dynamic.layout.esdc1a
    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    # Python layout: 8*g.n diff for Genrou (6 states + p_m0 + e_fd0),
    # then 3*exc.n for ESDC1A. Algebraic: 6*g.n (4 Genrou + p_m + e_fd alg).
    # Network: 2 entries per bus.
    py_diff_genrou = 8 * g.n
    py_alg_offset  = py_diff_genrou + 3 * exc.n     # 11 for this case
    py_net_offset  = py_alg_offset + 6 * g.n        # 17 for this case

    GROU_NAMES = ["e_qp","e_dp","phi_1d","phi_2q","w","delta"]
    GALG_NAMES = ["v_q","v_d","i_q","i_d"]
    EXC_NAMES  = ["vr1","vr2","e_fd"]

    rows = Tuple{String,Int,Int}[]
    for k in 1:g.n
        py_diff_base = (k-1)*8
        py_alg_base  = py_alg_offset + (k-1)*6
        jl_diff_base = Int(g.diff_ptr[k]) - 1
        jl_alg_base  = diff_dim + Int(g.alg_ptr[k]) - 1
        for (j, name) in enumerate(GROU_NAMES)
            push!(rows, ("g$k.$name", jl_diff_base + j, py_diff_base + j))
        end
        for (j, name) in enumerate(GALG_NAMES)
            push!(rows, ("g$k.$name", jl_alg_base + j, py_alg_base + j))
        end
    end
    for k in 1:exc.n
        py_diff_base = py_diff_genrou + (k-1)*3
        jl_diff_base = Int(exc.diff_ptr[k]) - 1
        for (j, name) in enumerate(EXC_NAMES)
            push!(rows, ("exc$k.$name", jl_diff_base + j, py_diff_base + j))
        end
    end
    net_jl = diff_dim + alg_dim
    for b in 1:length(ps.buses)
        push!(rows, ("bus$b.vr", net_jl + 2*(b-1) + 1, py_net_offset + 2*(b-1) + 1))
        push!(rows, ("bus$b.vi", net_jl + 2*(b-1) + 2, py_net_offset + 2*(b-1) + 2))
    end

    worst = 0.0; worst_name = ""; per_state = Dict{String,Float64}()
    for (name, jr, pr) in rows
        d = maximum(abs, traj[jr, :] .- hist[pr, :])
        per_state[name] = d
        if d > worst; worst = d; worst_name = name; end
    end

    passed = worst <= TOL
    criteria = Any[
        Dict("name" => "max_traj_err_inf", "value" => worst,
             "threshold" => TOL, "passed" => passed),
    ]
    metadata = Dict{String,Any}("wallclock_s" => time() - t0,
                                 "worst_state" => worst_name,
                                 "per_state_max_abs" => per_state,
                                 "reference_path" => REF,
                                 "git_sha" => git_sha(REPO))
    write_artifact(OUT, 3, "G4", passed, criteria, metadata)
    @printf "Phase 3 G4: passed=%s worst=%.3e (tol=%.1e) at %s -> %s\n" passed worst TOL worst_name OUT
    return passed
end

exit(main() ? 0 : 1)
