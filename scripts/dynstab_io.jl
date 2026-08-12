# Shared IO + scenario helpers for the dynstab generation pipeline.

using GradPower
using HDF5
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
resolve_path(p) = isabspath(p) ? p : joinpath(REPO_ROOT, p)

function get_rank_size()
    for (rk, sk) in (("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
                     ("PMI_RANK",             "PMI_SIZE"),
                     ("SLURM_PROCID",         "SLURM_NTASKS"),
                     ("RANK",                 "WORLD_SIZE"))
        if haskey(ENV, rk)
            rank = parse(Int, ENV[rk])
            size = haskey(ENV, sk) ? parse(Int, ENV[sk]) : 1
            return rank, size
        end
    end
    return 0, 1
end


struct Scenario
    case_idx::Int
    fault_bus_internal::Int
    fault_bus_ext::Int
    r_fault::Float64
    t_on::Float64
    t_off::Float64
end

function build_system(case)
    raw = resolve_path(case["raw"])
    dyr = resolve_path(case["dyr"])
    sys = GradPower.from_psse(raw, dyr)
    GradPower.build_network!(sys)
    GradPower.runpf!(sys; verbose=false)
    return sys
end


function fault_bus_pairs(sys, buses_cfg)
    pairs = Tuple{Int,Int}[]
    if buses_cfg isa AbstractString && lowercase(buses_cfg) == "all"
        for (internal, b) in enumerate(sys.buses)
            push!(pairs, (internal, b.i))
        end
    else

        for ext in buses_cfg
            ext_i = Int(ext)
            if haskey(sys.busmap, ext_i)
                push!(pairs, (sys.busmap[ext_i], ext_i))
            else
                @warn "Fault bus $ext_i not found in case; skipping."
            end
        end
    end
    return pairs
end

function enumerate_scenarios(cfg, systems)
    fault = cfg["fault"]
    r_faults = Float64.(fault["r_fault"])
    durations = Float64.(fault["durations"])
    t_on = Float64(fault["t_on"])
    scenarios = Scenario[]
    for (ci, case) in enumerate(cfg["cases"])
        for (internal, ext) in fault_bus_pairs(systems[ci], fault["buses"])
            for rf in r_faults, dur in durations
                push!(scenarios, Scenario(ci, internal, ext, rf, t_on, t_on + dur))
            end
        end
    end
    return scenarios
end


function write_grid!(fid, sys)
    g = dynamics_graph(sys)
    grp = create_group(fid, "grid")
    for (k, v) in g
        if v isa AbstractArray
            grp[k] = v
        else
            attributes(grp)[k] = v
        end
    end
    return g
end

function write_series!(parent, name, mat::AbstractMatrix{Float32})
    n, T = size(mat)
    chunk = (n, min(T, 256))
    if n == 0
        parent[name] = mat
        return
    end
    d = create_dataset(parent, name, Float32, (n, T);
                       chunk=chunk, shuffle=true, deflate=4)
    write(d, mat)
end

# Note: If the Lumina ingestion schema changes, this also needs to be updated.
function write_sample!(fid, idx::Int, sc::Scenario, ch, metrics)
    grp = create_group(fid, @sprintf("samples/%06d", idx))
    write_series!(grp, "bus_vm",    ch["bus_vm"])
    write_series!(grp, "bus_va",    ch["bus_va"])
    write_series!(grp, "gen_delta", ch["gen_delta"])
    write_series!(grp, "gen_omega", ch["gen_omega"])

    a = attributes(grp)
    a["fault_bus_internal"] = sc.fault_bus_internal
    a["fault_bus_ext"]      = sc.fault_bus_ext
    a["r_fault"]            = sc.r_fault
    a["t_on"]               = sc.t_on
    a["t_off"]              = sc.t_off
    a["clearing_time"]      = sc.t_off - sc.t_on
    a["T"]                  = ch["T"]
    for (k, v) in metrics
        a[k] = v isa Bool ? Int(v) : v   # HDF5 attrs: store bool as 0/1
    end
end
