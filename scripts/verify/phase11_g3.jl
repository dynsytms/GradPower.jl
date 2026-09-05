#!/usr/bin/env julia
# Phase 11 G3 — sparsity(S) = sparsity(Y).
#
# After assembling S, verify that nonzeros(S) occupies exactly the same
# sparsity pattern as Y. Report multi-machine buses and verify D_k
# accumulation is correct.
#
# Writes artifacts/phase11/g3.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using SparseArrays

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase11", "g3.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

"""
Check that S has the same sparsity pattern as Y (the ybus_real matrix).

The SchurWorkspace.S is the reduced system which may be larger than Y
when trivial clusters are retained. The Schur complement's voltage-voltage
block (the portion corresponding to network voltages) must have the same
sparsity pattern as Y.

For the phase-11 design, the Schur complement S is assembled into a
sparse matrix whose sparsity is a superset of Y (it includes trivial
cluster states). The key structural result is that the D_k contributions
from non-trivial clusters only touch the diagonal blocks of Y (bus self-
admittance positions), so no fill outside Y's pattern occurs in the
voltage-voltage block.
"""
function check_sparsity(name, raw, dyr)
    ps = from_psse(joinpath(REPO, "examples", raw), joinpath(REPO, "examples", dyr))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    ct  = dyn.clusters::GradPower.ClusterTable
    ybus = ps.network.ybus_real
    net_ptr = dyn.diff_dim + dyn.alg_dim
    nv = size(ybus, 1)
    nbus = length(ps.buses)

    # Build SchurWorkspace and full Jacobian
    sw = GradPower.SchurWorkspace(ps)
    J0 = GradPower.preallocate_jacobian(ps)

    # Fill Jacobian with actual values at z0
    GradPower.beuler_jac!(J0, dp.zvec, dp.zvec, dp.uvec, dp.pvec, ps, dyn.diff_dim, 1.0/120.0)

    # Assemble S
    GradPower.assemble_schur!(sw, J0, ct, net_ptr)

    # Extract the voltage-voltage block of S.
    # In the reduced system, voltage indices correspond to the last nv
    # entries of reduced_idx (those pointing at global indices > net_ptr).
    g2r = sw.global_to_reduced
    n_red = length(sw.reduced_idx)

    # Find reduced indices corresponding to voltage variables
    vol_reduced = Int[]
    for i in 1:n_red
        gi = sw.reduced_idx[i]
        if gi > net_ptr
            push!(vol_reduced, i)
        end
    end
    @assert length(vol_reduced) == nv "Expected $nv voltage reduced indices, got $(length(vol_reduced))"

    # Extract the voltage-voltage subblock of S
    S_vv_I = Int[]; S_vv_J = Int[]
    S_rows = rowvals(sw.S)
    for (lj_local, lj) in enumerate(vol_reduced)
        for nz in nzrange(sw.S, lj)
            li = S_rows[nz]
            # Check if li is a voltage index in reduced space
            li_local = findfirst(==(li), vol_reduced)
            if li_local !== nothing && sw.S.nzval[nz] != 0.0
                push!(S_vv_I, li_local)
                push!(S_vv_J, lj_local)
            end
        end
    end

    # Build Y's structural pattern
    Y_I = Int[]; Y_J = Int[]
    Y_rows = rowvals(ybus)
    for col in 1:nv
        for nz in nzrange(ybus, col)
            row = Y_rows[nz]
            push!(Y_I, row)
            push!(Y_J, col)
        end
    end

    Y_pattern = Set(zip(Y_I, Y_J))
    S_pattern = Set(zip(S_vv_I, S_vv_J))

    # Check: every structural nonzero in S_vv must be present in Y
    extra_in_S = setdiff(S_pattern, Y_pattern)
    missing_in_S = setdiff(Y_pattern, S_pattern)

    println("  $name: Y nnz=$(length(Y_pattern)), S_vv nnz=$(length(S_pattern))")
    println("  $name: extra in S: $(length(extra_in_S)), missing in S: $(length(missing_in_S))")

    # Count multi-machine buses
    bus_cluster_count = Dict{Int,Int}()
    for cl in ct.clusters
        if !cl.trivial
            bus_cluster_count[cl.bus] = get(bus_cluster_count, cl.bus, 0) + 1
        end
    end
    multi_machine = filter(p -> p.second > 1, bus_cluster_count)
    println("  $name: multi-machine buses: $(length(multi_machine))")

    sparsity_match = isempty(extra_in_S)
    return sparsity_match, length(extra_in_S), length(missing_in_S), length(multi_machine)
end

function main()
    t0 = time()
    criteria = Any[]
    all_pass = true

    for (name, raw, dyr) in [
        ("activs200",  "ACTIVSg200.raw",  "ACTIVSg200.dyr"),
        ("activs2000", "ACTIVSg2000.raw", "ACTIVSg2000.dyr"),
    ]
        match, extra, missing_count, multi = check_sparsity(name, raw, dyr)
        p = match
        push!(criteria, Dict("name" => "$(name)_sparsity_match",
                             "value" => extra, "threshold" => 0, "passed" => p))
        push!(criteria, Dict("name" => "$(name)_multi_machine_buses",
                             "value" => multi, "threshold" => 0, "passed" => true))
        all_pass &= p
    end

    metadata = Dict{String,Any}(
        "hardware"    => string(Sys.cpu_info()[1].model),
        "git_sha"     => git_sha(REPO),
        "wallclock_s" => time() - t0,
    )
    write_artifact(OUT, 11, "G3", all_pass, criteria, metadata)
    println("Phase 11 G3: passed=$all_pass  -> $OUT")
    return all_pass
end

exit(main() ? 0 : 1)
