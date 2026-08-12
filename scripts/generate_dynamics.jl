#!/usr/bin/env julia
# Dynamic-stability data generation driver.
#
# Reads a data generation config, enumerates fault scenarios, shards them round-robin across MPI ranks, solves with GradPower
# writes compressed HDF5 consumed in a schema compatible with Lumina
#
# Usage:
#   julia --project=scripts scripts/generate_dynamics.jl <config.toml> [--dry-run]
#
# Parallelism: launch N copies (e.g. `srun -n N julia ...`); each derives its
# (rank, size) from the environment (mirrors lumina's get_rank_size) and takes
# scenarios[rank+1 : size : end].

# Output layout:
#   <output_root>/<case>/<case>_rank<RRRR>.h5
#     attrs: schema_version, case_name, baseMVA, dt, dt_eff, t_final,
#            downsample, n_bus, n_gen, n_branch, n_samples, ...
#     /grid/*            static topology (written once per file)
#     /samples/<idx>/    one fault simulation each
#         attrs: fault_bus_ext, fault_bus_internal, r_fault, t_on, t_off,
#                clearing_time, stable, max_angle_sep_deg, max_freq_dev, ...
#         bus_vm,bus_va  [n_bus, T] f32   (chunked + gzip + shuffle)
#         gen_delta,gen_omega [n_gen, T] f32

using GradPower
using HDF5
using Printf
import TOML
import Dates

include(joinpath(@__DIR__, "dynstab_io.jl"))


function main()
    if isempty(ARGS)
        println("usage: julia --project=scripts scripts/generate_dynamics.jl <config.toml> [--dry-run]")
        return
    end
    config_path = ARGS[1]
    dry_run = "--dry-run" in ARGS
    cfg = TOML.parsefile(config_path)

    rank, world = get_rank_size()

    out_root = haskey(ENV, "DYNSTAB_OUTPUT_ROOT") ?
        ENV["DYNSTAB_OUTPUT_ROOT"] : resolve_path(cfg["output_root"])
    sim = cfg["sim"]
    dt = Float64(sim["dt"])
    t_final = Float64(sim["t_final"])
    downsample = Int(get(sim, "downsample", 1))
    lbl = get(cfg, "label", Dict())
    angle_thr = deg2rad(Float64(get(lbl, "angle_sep_threshold_deg", 180.0)))
    settle_ratio = Float64(get(lbl, "settle_ratio", 0.5))

    rank == 0 && @info "Building $(length(cfg["cases"])) case system(s)..."
    systems = [build_system(case) for case in cfg["cases"]]
    scenarios = enumerate_scenarios(cfg, systems)

    # round-robin shard index is scenarios[rank+1 : world : end]
    my_scenarios = scenarios[(rank + 1):world:length(scenarios)]

    if rank == 0
        @info "Total scenarios: $(length(scenarios)) across $world rank(s); ~$(length(my_scenarios)) on rank 0."
    end
    if dry_run
        rank == 0 && @info "Dry run: not simulating. Per-case scenario counts:"
        if rank == 0
            for (ci, case) in enumerate(cfg["cases"])
                n = count(s -> s.case_idx == ci, scenarios)
                println("  $(case["name"]): $n scenarios")
            end
        end
        return
    end

    by_case = Dict{Int,Vector{Scenario}}()
    for sc in my_scenarios
        push!(get!(by_case, sc.case_idx, Scenario[]), sc)
    end

    t_start = Dates.now()
    for (ci, scs) in by_case
        case = cfg["cases"][ci]
        name = case["name"]
        sys = systems[ci]
        case_dir = joinpath(out_root, name)
        mkpath(case_dir)
        out_file = joinpath(case_dir, @sprintf("%s_rank%04d.h5", name, rank))

        # Note: If the lumina heterodata schemas change this export format needs to be updated accordingly:
        h5open(out_file, "w") do fid
            ra = attributes(fid)
            ra["schema_version"] = EXPORT_SCHEMA_VERSION
            ra["case_name"]      = name
            ra["baseMVA"]        = sys.baseMVA
            ra["dt"]             = dt
            ra["dt_eff"]         = dt * downsample
            ra["t_final"]        = t_final
            ra["downsample"]     = downsample
            ra["n_bus"]          = length(sys.buses)
            ra["diff_dim"]       = sys.dynamic.diff_dim
            ra["alg_dim"]        = sys.dynamic.alg_dim
            ra["rank"]           = rank
            ra["world_size"]     = world

            write_grid!(fid, sys)

            nwritten = 0
            for (k, sc) in enumerate(scs)
                empty!(sys.dynamic.events)
                dprob = GradPower.DynamicProblem(sys)
                GradPower.initialize_dynamics!(dprob, sys)
                GradPower.add_event!(sys,
                    GradPower.ContingencyEvent(sc.fault_bus_internal, sc.r_fault, sc.t_on, sc.t_off))
                tvec, traj = GradPower.integrate!(dprob, sys, t_final; dt=dt)

                ch = dynamics_channels(sys, traj; downsample=downsample)
                metrics = stability_metrics(sys, traj;
                    angle_sep_threshold=angle_thr, settle_ratio=settle_ratio)
                write_sample!(fid, k, sc, ch, metrics)
                nwritten += 1
            end
            ra["n_samples"] = nwritten
            @info "[rank $rank] $name: wrote $nwritten samples -> $out_file"
        end
    end
    rank == 0 && @info "Done in $(Dates.canonicalize(Dates.now() - t_start))."
end

main()
