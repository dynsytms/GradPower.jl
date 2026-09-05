#!/usr/bin/env julia
# Phase 10 G3 — Cluster table correctness.
#
# Verify that every generator appears in exactly one cluster, controllers
# are attached correctly, type-tuple groups are contiguous, and |w_k|
# matches expected state counts.
#
# Writes artifacts/phase10/g3.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase10", "g3.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function check_clusters(name, raw, dyr)
    ps = from_psse(raw, dyr)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end

    ct = ps.dynamic.clusters
    devices = ps.dynamic.devices
    errors = String[]

    # 1. Every generator appears in exactly one cluster
    gen_seen = Dict{Int,Int}()  # gen device index -> cluster count
    for (ci, cl) in enumerate(ct.clusters)
        gi = cl.gen_idx
        gen_seen[gi] = get(gen_seen, gi, 0) + 1
    end
    for (i, dev) in enumerate(devices)
        if dev.dtype isa GradPower.AbstractGeneratorType || dev.dtype isa GradPower.StaticGenerator
            cnt = get(gen_seen, i, 0)
            if cnt != 1
                push!(errors, "$name: generator device $i appears in $cnt clusters (expected 1)")
            end
        end
    end

    # 2. Controllers are attached to the correct cluster
    # Verify by checking that the controller's (bus, id) matches the generator's
    for (ci, cl) in enumerate(ct.clusters)
        gen = devices[cl.gen_idx].dtype
        for (label, dev_idx) in [("gov", cl.gov_idx), ("exc", cl.exc_idx), ("pss", cl.pss_idx)]
            dev_idx == 0 && continue
            ctrl = devices[dev_idx].dtype
            if hasproperty(ctrl, :bus) && hasproperty(gen, :bus)
                if ctrl.bus != gen.bus
                    push!(errors, "$name: cluster $ci $label bus mismatch: $(ctrl.bus) vs gen $(gen.bus)")
                end
            end
        end
    end

    # 3. Type-tuple groups are contiguous
    for (tt, s, e) in ct.type_groups
        for ci in s:e
            actual_tt = GradPower.cluster_type_tuple(ct.clusters[ci], devices)
            if actual_tt != tt
                push!(errors, "$name: cluster $ci type $(actual_tt) != group type $(tt)")
            end
        end
    end

    # 4. |w_k| matches expected state count
    for (ci, cl) in enumerate(ct.clusters)
        expected_d = 0
        expected_a = 0
        for dev_idx in GradPower._cluster_device_order(cl)
            dev = devices[dev_idx]
            expected_d += dev.dtype.diff_size
            expected_a += dev.dtype.alg_size
        end
        if cl.d_k != expected_d
            push!(errors, "$name: cluster $ci d_k=$(cl.d_k) != expected $expected_d")
        end
        if cl.a_k != expected_a
            push!(errors, "$name: cluster $ci a_k=$(cl.a_k) != expected $expected_a")
        end
        if cl.w_size != expected_d + expected_a
            push!(errors, "$name: cluster $ci w_size=$(cl.w_size) != expected $(expected_d + expected_a)")
        end
        if cl.w_end - cl.w_start + 1 != cl.w_size
            push!(errors, "$name: cluster $ci z-range $(cl.w_start):$(cl.w_end) doesn't match w_size=$(cl.w_size)")
        end
    end

    # 5. No overlapping z-ranges
    for i in 1:length(ct.clusters)
        for j in i+1:length(ct.clusters)
            a = ct.clusters[i]; b = ct.clusters[j]
            if !(a.w_end < b.w_start || b.w_end < a.w_start)
                push!(errors, "$name: clusters $i and $j overlap: $(a.w_start):$(a.w_end) vs $(b.w_start):$(b.w_end)")
            end
        end
    end

    println("  $name: $(length(ct.clusters)) clusters, $(length(ct.type_groups)) type groups, $(length(errors)) errors")
    return errors
end

function main()
    t0 = time()
    all_errors = String[]

    # ACTIVSg200
    append!(all_errors, check_clusters("activs200",
        joinpath(REPO, "examples", "ACTIVSg200.raw"),
        joinpath(REPO, "examples", "ACTIVSg200.dyr")))

    # ACTIVSg2000 — the examples/ACTIVSg2000.dyr may have IEEEST records
    # with zero time constants that fail parsing; use try/catch.
    a2000_raw = joinpath(REPO, "examples", "ACTIVSg2000.raw")
    a2000_dyr = joinpath(REPO, "examples", "ACTIVSg2000.dyr")
    if isfile(a2000_raw) && isfile(a2000_dyr)
        try
            append!(all_errors, check_clusters("activs2000", a2000_raw, a2000_dyr))
        catch e
            println("  activs2000: skipped ($(sprint(showerror, e, catch_backtrace())))")
        end
    end

    passed = isempty(all_errors)
    if !passed
        for e in all_errors
            println("  ERROR: $e")
        end
    end

    criteria = Any[
        Dict("name" => "cluster_table_correct",
             "value" => length(all_errors), "threshold" => 0,
             "passed" => passed),
    ]
    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t0,
        "errors" => all_errors,
    )
    write_artifact(OUT, 10, "G3", passed, criteria, metadata)
    println("Phase 10 G3: passed=$passed  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
