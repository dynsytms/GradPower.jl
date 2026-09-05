#!/usr/bin/env julia
# Phase 10 G4 — A_k extraction matches global Jacobian.
#
# For every cluster k, verify that extract_Ak! output matches the
# corresponding |w_k| x |w_k| submatrix of the global sparse Jacobian.
# Also verify extract_Bk_Ck!.
#
# Writes artifacts/phase10/g4.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using SparseArrays

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase10", "g4.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function main()
    t0 = time()

    raw = joinpath(REPO, "examples", "ACTIVSg200.raw")
    dyr = joinpath(REPO, "examples", "ACTIVSg200.dyr")
    ps = from_psse(raw, dyr)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    # Evaluate Jacobian at z0
    J = GradPower.preallocate_jacobian(ps)
    GradPower.rhs_jac!(J, dp.zvec, dp.uvec, dp.pvec, ps)

    ct = ps.dynamic.clusters
    net_ptr = ps.dynamic.diff_dim + ps.dynamic.alg_dim

    max_Ak_err = 0.0
    max_Bk_err = 0.0
    max_Ck_err = 0.0

    for (ci, cl) in enumerate(ct.clusters)
        wk = cl.w_size
        ws = cl.w_start
        we = cl.w_end

        # Extract A_k
        A_dense = zeros(wk, wk)
        GradPower.extract_Ak!(A_dense, J, cl)

        # Reference: direct submatrix from J
        A_ref = Matrix(J[ws:we, ws:we])
        err_A = maximum(abs, A_dense .- A_ref)
        if err_A > max_Ak_err
            max_Ak_err = err_A
        end

        # Extract B_k, C_k
        B_tilde = zeros(wk, 2)
        C_tilde = zeros(2, wk)
        GradPower.extract_Bk_Ck!(B_tilde, C_tilde, J, cl, net_ptr)

        # Reference B_tilde: columns of J at (w_k rows, vr/vi columns)
        vr_col = net_ptr + 2*(cl.bus - 1) + 1
        vi_col = vr_col + 1
        B_ref = Matrix(J[ws:we, [vr_col, vi_col]])
        err_B = maximum(abs, B_tilde .- B_ref)
        if err_B > max_Bk_err
            max_Bk_err = err_B
        end

        # Reference C_tilde: rows of J at (vr/vi rows, w_k columns)
        C_ref = Matrix(J[[vr_col, vi_col], ws:we])
        err_C = maximum(abs, C_tilde .- C_ref)
        if err_C > max_Ck_err
            max_Ck_err = err_C
        end
    end

    pass_A = max_Ak_err <= 1e-15
    pass_B = max_Bk_err <= 1e-15
    pass_C = max_Ck_err <= 1e-15
    passed = pass_A && pass_B && pass_C

    criteria = Any[
        Dict("name" => "max_Ak_err", "value" => max_Ak_err,
             "threshold" => 1e-15, "passed" => pass_A),
        Dict("name" => "max_Bk_err", "value" => max_Bk_err,
             "threshold" => 1e-15, "passed" => pass_B),
        Dict("name" => "max_Ck_err", "value" => max_Ck_err,
             "threshold" => 1e-15, "passed" => pass_C),
    ]
    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t0,
        "n_clusters" => length(ct.clusters),
    )
    write_artifact(OUT, 10, "G4", passed, criteria, metadata)
    println("Phase 10 G4: passed=$passed  Ak_err=$max_Ak_err  Bk_err=$max_Bk_err  Ck_err=$max_Ck_err  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
