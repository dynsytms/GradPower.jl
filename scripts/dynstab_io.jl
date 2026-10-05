# Shared IO + scenario helpers for the dynstab generation pipeline.
#
# Included by BOTH scripts/generate_dynamics.jl (production driver) and
# benchmarks/bench_generate.jl (instrumentation) so the benchmark times the
# exact code paths that production runs. Assumes `using GradPower, HDF5, Printf`
# have been (or are) loaded; the `using` lines below are idempotent.

using GradPower
using HDF5
using Printf

# Resolve a path relative to the repo root (parent of scripts/) unless absolute.
# @__DIR__ is this file's directory (scripts/) regardless of the includer.
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
resolve_path(p) = isabspath(p) ? p : joinpath(REPO_ROOT, p)

# Operating-point sampling + admission (build_op_system, op_config, ...).
include(joinpath(@__DIR__, "operating_points.jl"))

# --------------------------------------------------------------------------
# Rank/size discovery — env-var based, mirrors lumina scripts/data_process.py
# `get_rank_size()`. Returns 0-based rank and world size.
# --------------------------------------------------------------------------
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

# contiguous block `rank` of 1:n split into `world` near-equal parts
function _block(n, rank, world)
    q, r = divrem(n, world)
    lo = rank * q + min(rank, r) + 1
    hi = lo + q - 1 + (rank < r ? 1 : 0)
    return lo:hi
end

# `--flag value` from ARGS (as a String), or `default`
_arg(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i + 1])

# --------------------------------------------------------------------------
# Scenario enumeration
# --------------------------------------------------------------------------
# A scenario is one simulated disturbance. `kind` selects the fault family:
#
#   :bus_fault        3-phase shunt fault at a bus, self-clearing at t_off.
#   :line_trip        a branch is removed at t_on (permanent if reclose==Inf,
#                     otherwise restored at `reclose`). No bus fault.
#   :fault_plus_trip  the realistic protection sequence: shunt fault at a bus
#                     from t_on to t_off, cleared BY tripping one of the lines
#                     incident to that bus at t_off. Permanent unless
#                     `reclose` is finite.
#
# Fields not relevant to a kind are 0 (buses/lines) or Inf (reclose).
# `op_id` names the operating point (0 = the case as published; see
# operating_points.jl).
struct Scenario
    case_idx::Int
    kind::Symbol
    fault_bus_internal::Int
    fault_bus_ext::Int
    r_fault::Float64
    t_on::Float64
    t_off::Float64
    line_fr_ext::Int
    line_to_ext::Int
    reclose::Float64
    op_id::Int
end

# Backward-compatible constructors: the published operating point.
Scenario(case_idx, kind::Symbol, fbi, fbe, rf, ton, toff, lfr, lto, rc) =
    Scenario(case_idx, kind, fbi, fbe, rf, ton, toff, lfr, lto, rc, 0)
Scenario(case_idx, fbi, fbe, rf, ton, toff) =
    Scenario(case_idx, :bus_fault, fbi, fbe, rf, ton, toff, 0, 0, Inf, 0)

const FAULT_FAMILIES = (:bus_fault, :line_trip, :fault_plus_trip)

# Per-case `static_gen_mode` for units without a dynamic model: "pv" (default;
# they regulate their bus voltage with unlimited Q), "pq" (hold power-flow P
# and Q) or "load" (netted as negative constant-admittance loads).
static_gen_mode(case) = Symbol(lowercase(get(case, "static_gen_mode", "pv")))

function build_system(case)
    raw = resolve_path(case["raw"])
    dyr = resolve_path(case["dyr"])
    sys = GradPower.from_psse(raw, dyr; static_gen_mode=static_gen_mode(case))
    GradPower.build_network!(sys)
    GradPower.runpf!(sys; verbose=false)
    return sys
end

# Map the configured fault buses to (internal, external) index pairs.
# "all" means every PHYSICAL bus: the star points that raw_to_grad synthesises
# for three-winding transformers (id "STAR") are not substations and are
# skipped.
function fault_bus_pairs(sys, buses_cfg)
    pairs = Tuple{Int,Int}[]
    if buses_cfg isa AbstractString && lowercase(buses_cfg) == "all"
        for (internal, b) in enumerate(sys.buses)
            b.id == "STAR" && continue
            push!(pairs, (internal, b.i))
        end
    else
        # list of EXTERNAL PSS/E bus numbers
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

# Bridges of the network graph, as a set of unordered INTERNAL (fr,to) pairs.
#
# Removing a bridge disconnects the graph, i.e. it islands part of the system.
# On IEEE9 the bridges are exactly the three generator step-up connections
# (1-4, 2-7, 9-3): tripping one strands a machine with no load, which shows up
# in the labels as a colossal angle spread. That is islanding, not transient
# instability, so the sweep can optionally skip these (`skip_islanding = true`, the default).

function network_bridges(sys)
    n = length(sys.buses)
    adj = [Int[] for _ in 1:n]
    multiplicity = Dict{Tuple{Int,Int},Int}()
    for br in sys.branches
        br.fr == br.to && continue
        key = br.fr < br.to ? (br.fr, br.to) : (br.to, br.fr)
        multiplicity[key] = get(multiplicity, key, 0) + 1
    end
    for ((u, v), _) in multiplicity
        push!(adj[u], v); push!(adj[v], u)
    end

    disc = zeros(Int, n); low = zeros(Int, n); timer = 0
    bridges = Set{Tuple{Int,Int}}()
    for root in 1:n
        disc[root] != 0 && continue
        # stack entries: (node, parent, next-neighbour-index)
        stack = Tuple{Int,Int,Int}[(root, 0, 1)]
        timer += 1; disc[root] = timer; low[root] = timer
        while !isempty(stack)
            (u, parent, ni) = pop!(stack)
            if ni <= length(adj[u])
                push!(stack, (u, parent, ni + 1))
                v = adj[u][ni]
                v == parent && continue
                if disc[v] == 0
                    timer += 1; disc[v] = timer; low[v] = timer
                    push!(stack, (v, u, 1))
                else
                    low[u] = min(low[u], disc[v])
                end
            else
                if parent != 0
                    low[parent] = min(low[parent], low[u])
                    if low[u] > disc[parent]
                        key = parent < u ? (parent, u) : (u, parent)
                        multiplicity[key] == 1 && push!(bridges, key)
                    end
                end
            end
        end
    end
    return bridges
end

# Branches to consider for the :line_trip / :fault_plus_trip families.
# `lines_cfg` is "all" (every branch) or a list of [from_ext, to_ext] pairs.
# Returns (from_ext, to_ext) pairs; self-loops and parallel duplicates are
# collapsed so a branch is only enumerated once.
function line_pairs(sys, lines_cfg; skip_islanding::Bool=true)
    ext_of = Dict(v => k for (k, v) in sys.busmap)
    bridges = skip_islanding ? network_bridges(sys) : Set{Tuple{Int,Int}}()
    _islands(fr_i, to_i) = (fr_i < to_i ? (fr_i, to_i) : (to_i, fr_i)) in bridges
    out = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    if lines_cfg isa AbstractString && lowercase(lines_cfg) == "all"
        for br in sys.branches
            br.fr == br.to && continue
            _islands(br.fr, br.to) && continue
            fr = get(ext_of, br.fr, 0); to = get(ext_of, br.to, 0)
            (fr == 0 || to == 0) && continue
            key = fr < to ? (fr, to) : (to, fr)
            key in seen && continue
            push!(seen, key); push!(out, (fr, to))
        end
    else
        for pr in lines_cfg
            fr = Int(pr[1]); to = Int(pr[2])
            if !haskey(sys.busmap, fr) || !haskey(sys.busmap, to)
                @warn "Trip line ($fr,$to) not found in case; skipping."
                continue
            end
            if _islands(sys.busmap[fr], sys.busmap[to])
                @info "Trip line ($fr,$to) is a bridge (islands the network); skipping."
                continue
            end
            key = fr < to ? (fr, to) : (to, fr)
            key in seen && continue
            push!(seen, key); push!(out, (fr, to))
        end
    end
    return out
end

# Branches incident to a given internal bus index, as (from_ext, to_ext).
function incident_line_pairs(sys, bus_internal::Int; bridges=nothing)
    ext_of = Dict(v => k for (k, v) in sys.busmap)
    brs = bridges === nothing ? Set{Tuple{Int,Int}}() : bridges
    out = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for br in sys.branches
        (br.fr == bus_internal || br.to == bus_internal) || continue
        br.fr == br.to && continue
        ((br.fr < br.to ? (br.fr, br.to) : (br.to, br.fr)) in brs) && continue
        fr = get(ext_of, br.fr, 0); to = get(ext_of, br.to, 0)
        (fr == 0 || to == 0) && continue
        key = fr < to ? (fr, to) : (to, fr)
        key in seen && continue
        push!(seen, key); push!(out, (fr, to))
    end
    return out
end

"""
    enumerate_scenarios(cfg, systems) -> Vector{Scenario}

Enumerate the full sweep across cases, operating points and fault families.
`cfg["fault"]["families"]` selects the families; it defaults to
`["bus_fault"]`, so a config that predates fault families (and operating
points) produces exactly the scenario list it always did.

    [fault]
    families      = ["bus_fault", "line_trip", "fault_plus_trip"]
    buses         = "all"         # or a list of external bus numbers
    buses_per_op  = 0             # >0: random subset of `buses` per operating point
    r_fault       = [0.05]
    t_on          = 0.2
    durations     = [0.1]

    [fault.line_trip]
    lines         = "all"         # or [[from_ext, to_ext], ...]
    lines_per_op  = 0             # >0: random subset per operating point
    reclose       = []            # [] = permanent; else list of reclose DELAYS (s)

    [fault.fault_plus_trip]
    lines         = "incident"    # only branches touching the faulted bus
    lines_per_bus = 0             # >0: random subset of those per faulted bus
    reclose       = []

Operating points come from `op_config` (see operating_points.jl): every case
is enumerated once per op_id, and the per-op subsets are drawn from a seeded
RNG keyed on (case, op), so every rank enumerates the identical list. The list
is ordered case → op → family, which lets the driver shard it in contiguous
blocks and build each operating point once per rank.
"""
function enumerate_scenarios(cfg, systems)
    fault = cfg["fault"]
    r_faults  = Float64.(fault["r_fault"])
    durations = Float64.(fault["durations"])
    t_on      = Float64(fault["t_on"])
    nbus_op   = Int(get(fault, "buses_per_op", 0))

    fams = Symbol.(get(fault, "families", ["bus_fault"]))
    for f in fams
        f in FAULT_FAMILIES ||
            error("Unknown fault family :$f — valid families are $(FAULT_FAMILIES)")
    end

    # choose k of v (all of v when k <= 0 or k >= length(v)), order preserved
    _subset(rng, v, k) = (k <= 0 || k >= length(v)) ? v : v[sort(randperm(rng, length(v))[1:k])]

    scenarios = Scenario[]
    for (ci, case) in enumerate(cfg["cases"])
        sys = systems[ci]
        opc = op_config(cfg, case)
        all_buses = fault_bus_pairs(sys, fault["buses"])

        lt = get(fault, "line_trip", Dict())
        lt_lines = :line_trip in fams ?
            line_pairs(sys, get(lt, "lines", "all");
                       skip_islanding = Bool(get(lt, "skip_islanding", true))) :
            Tuple{Int,Int}[]
        lt_recl = Float64.(get(lt, "reclose", Float64[]))
        isempty(lt_recl) && (lt_recl = [Inf])

        ft = get(fault, "fault_plus_trip", Dict())
        ft_lines_cfg = get(ft, "lines", "incident")
        ft_recl = Float64.(get(ft, "reclose", Float64[]))
        isempty(ft_recl) && (ft_recl = [Inf])
        ft_skip = Bool(get(ft, "skip_islanding", true))
        # Computed once per case — the DFS is O(V+E) but not free at 70k buses.
        brs = (:fault_plus_trip in fams && ft_skip) ? network_bridges(sys) : Set{Tuple{Int,Int}}()

        for op in op_ids(opc)
            # separate stream from the OP's physical parameters
            rng = op_rng(opc, ci, op + 7_000_000)
            bus_pairs = _subset(rng, all_buses, nbus_op)

            if :bus_fault in fams
                for (internal, ext) in bus_pairs, rf in r_faults, dur in durations
                    push!(scenarios, Scenario(ci, :bus_fault, internal, ext, rf,
                                              t_on, t_on + dur, 0, 0, Inf, op))
                end
            end

            if :line_trip in fams
                for (fr, to) in _subset(rng, lt_lines, Int(get(lt, "lines_per_op", 0))), rc in lt_recl
                    # No shunt fault: r_fault/t_off are not meaningful here.
                    push!(scenarios, Scenario(ci, :line_trip, 0, 0, NaN,
                                              t_on, t_on, fr, to,
                                              isfinite(rc) ? t_on + rc : Inf, op))
                end
            end

            if :fault_plus_trip in fams
                for (internal, ext) in bus_pairs
                    cand = (ft_lines_cfg isa AbstractString &&
                            lowercase(ft_lines_cfg) == "incident") ?
                           incident_line_pairs(sys, internal; bridges = brs) :
                           line_pairs(sys, ft_lines_cfg; skip_islanding = ft_skip)
                    cand = _subset(rng, cand, Int(get(ft, "lines_per_bus", 0)))
                    for (fr, to) in cand, rf in r_faults, dur in durations, rc in ft_recl
                        t_off = t_on + dur
                        push!(scenarios, Scenario(ci, :fault_plus_trip, internal, ext,
                                                  rf, t_on, t_off, fr, to,
                                                  isfinite(rc) ? t_off + rc : Inf, op))
                    end
                end
            end
        end
    end
    return scenarios
end

"""
    apply_scenario!(sys, sc)

Clear any previously scheduled events and install the ones this scenario
needs. Must be called AFTER `restore_network!` so the branch admittances are
pristine (see the note there).
"""
function apply_scenario!(sys, sc::Scenario)
    empty!(sys.dynamic.events)
    empty!(sys.dynamic.trip_events)
    empty!(sys.dynamic.disconnect_events)

    if sc.kind === :bus_fault || sc.kind === :fault_plus_trip
        GradPower.add_event!(sys,
            GradPower.ContingencyEvent(sc.fault_bus_internal, sc.r_fault,
                                       sc.t_on, sc.t_off))
    end

    if sc.kind === :line_trip
        GradPower.add_trip_event!(sys,
            GradPower.create_trip_line_event(sys, sc.line_fr_ext, sc.line_to_ext,
                                             sc.t_on; toff = sc.reclose))
    elseif sc.kind === :fault_plus_trip
        # The line is what CLEARS the fault: it opens exactly at t_off.
        GradPower.add_trip_event!(sys,
            GradPower.create_trip_line_event(sys, sc.line_fr_ext, sc.line_to_ext,
                                             sc.t_off; toff = sc.reclose))
    end
    return nothing
end

"""
    snapshot_network(sys) -> Vector{Float64}
    restore_network!(sys, snap)

`integrate!` applies a line trip by mutating `ps.network.ybus_real` in place
(`_apply_trip_line!`) and only undoes it if a reclose is scheduled. A
PERMANENT trip therefore leaves the admittance matrix modified after the
simulation returns, which would silently corrupt every later scenario run
against the same `PowerSystem`. The generation driver snapshots the nonzeros
once per operating point and restores them before each scenario.
"""
snapshot_network(sys) = copy(sys.network.ybus_real.nzval)

function restore_network!(sys, snap::Vector{Float64})
    copyto!(sys.network.ybus_real.nzval, snap)
    return nothing
end

# --------------------------------------------------------------------------
# HDF5 writers
# --------------------------------------------------------------------------
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

# chunked + shuffle + gzip dataset for a per-node time-series matrix.
function write_series!(parent, name, mat::AbstractMatrix{Float32})
    n, T = size(mat)
    chunk = (n, min(T, 256))
    if n == 0
        parent[name] = mat   # empty edge case: no chunking
        return
    end
    d = create_dataset(parent, name, Float32, (n, T);
                       chunk=chunk, shuffle=true, deflate=4)
    write(d, mat)
end

# Sample groups are named by the scenario's GLOBAL index in the enumerated
# sweep, so names are unique across ranks and a restarted rank can tell which
# of its scenarios are already on disk.
sample_key(idx::Int) = @sprintf("samples/%07d", idx)

function write_sample!(fid, idx::Int, sc::Scenario, ch, metrics)
    grp = create_group(fid, sample_key(idx))
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
    # Fault-family provenance. `family` distinguishes bus_fault / line_trip /
    # fault_plus_trip; line_* are 0 when no branch is involved and `reclose`
    # is Inf for a permanent trip.
    a["family"]             = String(sc.kind)
    a["line_fr_ext"]        = sc.line_fr_ext
    a["line_to_ext"]        = sc.line_to_ext
    a["reclose"]            = sc.reclose
    a["op_id"]              = sc.op_id
    a["scenario_idx"]       = idx
    a["T"]                  = ch["T"]
    for (k, v) in metrics
        a[k] = v isa Bool ? Int(v) : v   # HDF5 attrs: store bool as 0/1
    end
end

"""
    write_failure!(fid, idx, sc, err)

A scenario whose integration threw (e.g. Newton failed to converge after a
severe fault). Recorded under `/failed/<idx>` with the scenario attributes and
the error text, so failures are counted rather than lost; `/samples` keeps
only complete trajectories.
"""
function write_failure!(fid, idx::Int, sc::Scenario, msg::AbstractString)
    grp = create_group(fid, @sprintf("failed/%07d", idx))
    a = attributes(grp)
    a["family"] = String(sc.kind); a["op_id"] = sc.op_id
    a["fault_bus_ext"] = sc.fault_bus_ext; a["r_fault"] = sc.r_fault
    a["t_on"] = sc.t_on; a["t_off"] = sc.t_off
    a["line_fr_ext"] = sc.line_fr_ext; a["line_to_ext"] = sc.line_to_ext
    a["reclose"] = sc.reclose; a["scenario_idx"] = idx
    a["error"] = first(msg, 2000)
end

"""
    write_operating_point!(fid, sys, rec)

`/operating_points/<op_id>`: what differs between operating points of one
case — the solved bus voltages, loads, and machine dispatch (generator rows in
`/grid/gen_*` order) — plus every sampled parameter and admission metric in
`rec` as attributes. `/grid` keeps the topology and the published (op 0)
state. Rejected operating points are written too (attributes only).
"""
function write_operating_point!(fid, sys, rec)
    key = @sprintf("operating_points/%04d", rec["op_id"])
    haskey(fid, key) && return
    grp = create_group(fid, key)
    for (k, v) in rec
        attributes(grp)[k] = v isa Bool ? Int(v) : v
    end
    get(rec, "admitted", false) || return
    grp["bus_v0m"] = Float32[b.v0m for b in sys.buses]
    grp["bus_v0a"] = Float32[b.v0a for b in sys.buses]
    grp["load_pd"] = Float32[l.pd for l in sys.loads]
    grp["load_qd"] = Float32[l.qd for l in sys.loads]
    gn = GradPower._genrou_nodes(sys)
    gidx = [sys.dynamic.map.gen[findfirst(d -> d.diff_ptr == n.diff_ptr &&
                                          d.dtype isa GradPower.Genrou, sys.dynamic.devices)]
            for n in gn]
    grp["gen_pg"] = Float32[g > 0 ? sys.gens[g].psch : NaN for g in gidx]
    grp["gen_qg"] = Float32[g > 0 ? sys.gens[g].qsch : NaN for g in gidx]
end
