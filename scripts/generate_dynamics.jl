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
    residual_tol = Float64(get(sim, "residual_tol", 1e-6))
    check_stability = Bool(get(sim, "check_self_stability", true))
    lbl = get(cfg, "label", Dict())
    angle_thr = deg2rad(Float64(get(lbl, "angle_sep_threshold_deg", 180.0)))
    settle_ratio = Float64(get(lbl, "settle_ratio", 0.5))

    rank == 0 && @info "Building $(length(cfg["cases"])) case system(s)..."
    built   = [build_system(case) for case in cfg["cases"]]
    systems = first.(built)
    bases   = last.(built)
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
                lams = sort(unique(s.load_scale for s in scenarios if s.case_idx == ci))
                @printf("  %s: %d scenarios, %d load draw(s) in [%.4f, %.4f]\n",
                        case["name"], n, length(lams),
                        isempty(lams) ? NaN : minimum(lams), isempty(lams) ? NaN : maximum(lams))
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
        base = bases[ci]
        zip_alpha = Float64(get(case, "zip_alpha", 1.0))
        # Group this rank's work by operating point so the power flow and the
        # dynamics initialization are paid once per lambda, not once per fault.
        sort!(scs, by = s -> (s.load_draw, s.fault_bus_internal, s.r_fault, s.t_off))

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
            ra["zip_alpha"]      = zip_alpha
            ra["rank"]           = rank
            ra["world_size"]     = world

            write_grid!(fid, sys)

            nwritten = 0
            current_draw = -1
            dprob = nothing
            z0 = Float64[]
            for (k, sc) in enumerate(scs)
                if sc.load_draw != current_draw
                    # New operating point: rescale, re-solve the power flow, and
                    # re-initialize. initialized_problem asserts the equilibrium
                    # residual, so a stale operating point fails loudly here.
                    apply_load_scale!(sys, base, sc.load_scale; zip_alpha=zip_alpha,
                                      enforce_q_limits=q_limits_enabled(case))
                    empty!(sys.dynamic.events)
                    dprob, res = initialized_problem(sys; residual_tol=residual_tol)
                    z0 = copy(dprob.zvec)
                    current_draw = sc.load_draw
                    @info "[rank $rank] $name: load_draw=$(sc.load_draw) lambda=$(round(sc.load_scale, digits=5)) residual=$(res)"

                    # Preflight: confirm the equilibrium is actually stable, so
                    # we don't generate thousands of samples whose labels are
                    # set by a fault-independent unstable mode. See
                    # check_self_stability for why this is not hypothetical.
                    if check_stability
                        chk = check_self_stability(sys, dprob, z0; t_final=t_final, dt=dt)
                        chk.ok || error("""
                            $name at lambda=$(sc.load_scale) is NOT a stable equilibrium: a 1e-6 speed
                            kick with no fault grows to $(round(chk.sep_deg, digits=2)) deg. Labels from this case would
                            reflect that unstable mode, not the fault. Check that the .dyr's governor
                            and exciter models are actually supported by the parser (unsupported rows
                            are skipped with a warning at parse time). Set
                            [sim] check_self_stability = false to override.""")
                    end
                end

                empty!(sys.dynamic.events)
                dprob.zvec .= z0            # every fault starts from this operating point
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
