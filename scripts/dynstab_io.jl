# Shared IO + scenario helpers for the dynstab generation pipeline.

using GradPower
using HDF5
using Printf
using Random

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
resolve_path(p) = isabspath(p) ? p : joinpath(REPO_ROOT, p)

# Bump when the on-disk HDF5 layout changes in a way consumers must notice.
const EXPORT_SCHEMA_VERSION = 3

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


# A scenario is one simulation: an operating point (load_scale) plus a fault.
struct Scenario
    case_idx::Int
    load_scale::Float64      # lambda, sampled from the [load] distribution
    load_draw::Int           # which lambda draw this is (0-based), for grouping
    fault_bus_internal::Int
    fault_bus_ext::Int
    r_fault::Float64
    t_on::Float64
    t_off::Float64
end


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------

# The base (lambda = 1) operating point, captured once per case so that every
# lambda is applied to the ORIGINAL values rather than compounding.
#
# It must be captured BEFORE the first power flow, not after. Under Q-limit
# enforcement the power flow mutates `bus.type` (PV -> PQ) and `gen.qsch`
# (pinned at a bound), and those are exactly the fields each new lambda has to
# start clean from. Snapshotting them after a solve would make every subsequent
# lambda inherit the lambda = 1 active set instead of finding its own. That is
# why `build_system` returns the BasePoint itself -- there is no way to call it
# at the wrong moment.
struct BasePoint
    pd::Vector{Float64}        # per ps.loads
    qd::Vector{Float64}
    psch::Vector{Float64}      # per ps.gens
    qsch::Vector{Float64}
    bus_type::Vector{Int64}    # pre-Q-limit types; the loop converts PV->PQ
    vset::Vector{Float64}      # PV setpoints, released along with the type
    is_slack_gen::Vector{Bool}
    zip_pinj::Vector{Float64}  # per ZIPLoad device, in device order
    zip_qinj::Vector{Float64}
    zip_dev_idx::Vector{Int}
end

function capture_base(sys)
    slack = Set(i for (i, b) in enumerate(sys.buses) if b.type == 3)
    zi = Int[]; zp = Float64[]; zq = Float64[]
    for (i, dev) in enumerate(sys.dynamic.devices)
        if dev.dtype isa GradPower.ZIPLoad
            push!(zi, i); push!(zp, dev.dtype.pinj); push!(zq, dev.dtype.qinj)
        end
    end
    return BasePoint([l.pd for l in sys.loads], [l.qd for l in sys.loads],
                     [g.psch for g in sys.gens], [g.qsch for g in sys.gens],
                     [b.type for b in sys.buses], [b.v0m for b in sys.buses],
                     [g.bus in slack for g in sys.gens],
                     zp, zq, zi)
end

"""
    apply_load_scale!(sys, base, lambda; zip_alpha)

Set the operating point to `lambda` x base loading and re-solve the power flow.

Scales every load (P and Q) and every non-slack generator's scheduled P, then
re-runs the power flow so the slack picks up the mismatch. `enforce_q_limits`
must match the flag used for the base solve -- a different active set at a
different lambda is fine and expected, but silently dropping the limits at some
lambdas would make the sampled operating points incomparable. The ZIPLoad devices
that the dynamics actually integrate are re-synced afterwards:

  * `pinj`/`qinj` from the scaled static loads, and
  * `v0mag` from the NEW power-flow voltage.

Refreshing `v0mag` is not optional. It is the reference voltage of the ZIP
model, captured from the .raw file before the power flow ever ran. If it is
left stale, the load consumes something other than `pinj` at the new operating
voltage and the dynamics no longer start at an equilibrium.
"""
function apply_load_scale!(sys, base::BasePoint, lambda::Float64;
                           zip_alpha::Float64=1.0, enforce_q_limits::Bool=true)
    # Undo the previous lambda's Q-limit active set before re-solving. The
    # limit loop converts PV buses to PQ and pins their generators; leaving
    # that in place would make each lambda inherit the last one's active set
    # instead of finding its own.
    for (i, bus) in enumerate(sys.buses)
        bus.type = base.bus_type[i]
        base.bus_type[i] == 2 && (bus.v0m = base.vset[i])
    end
    for (g, gen) in enumerate(sys.gens)
        gen.qsch = base.qsch[g]
    end

    for (k, load) in enumerate(sys.loads)
        load.pd = base.pd[k] * lambda
        load.qd = base.qd[k] * lambda
    end
    for (g, gen) in enumerate(sys.gens)
        base.is_slack_gen[g] || (gen.psch = base.psch[g] * lambda)
    end

    GradPower.runpf!(sys; verbose=false, enforce_q_limits=enforce_q_limits)

    for (j, i) in enumerate(base.zip_dev_idx)
        zl = sys.dynamic.devices[i].dtype
        zl.pinj  = base.zip_pinj[j] * lambda
        zl.qinj  = base.zip_qinj[j] * lambda
        zl.v0mag = sys.buses[zl.bus].v0m
        zl.α     = zip_alpha
    end
    return sys
end

"""
    initialized_problem(sys; residual_tol)

Build and initialize a DynamicProblem, verifying it really is at equilibrium.
Returns `(dprob, residual)`. Throws if the residual exceeds `residual_tol`,
which is the guard against a silently stale operating point.
"""
function initialized_problem(sys; residual_tol::Float64=1e-6)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    f = zeros(length(dprob.zvec))
    GradPower.rhs_fun!(f, dprob.zvec, dprob.uvec, dprob.pvec, sys)
    res = maximum(abs, f)
    res <= residual_tol ||
        error("equilibrium residual $res exceeds tol $residual_tol; operating point is not consistent")
    return dprob, res
end


"""
    check_self_stability(sys, dprob, z0; t_final, dt, eps, tol_deg)

Preflight: is this operating point actually a STABLE equilibrium?

Applies an infinitesimal speed kick to one machine, runs with NO fault, and
measures the resulting angle separation. A usable case returns ~0 degrees.

This is not paranoia. ACTIVSg2000 passes every other check -- the power flow
solves, the initialization residual is ~1e-9, and a no-fault run is flat to
1e-10 -- yet a 1e-8 speed perturbation grows to 140 degrees in 5 s, because all
334 machines have D = 0 and the .dyr's governors (GGOV1, IEEEG1, HYGOV) and
most of its exciters are model types this parser does not implement, so they
are silently skipped. The trajectory is then dominated by a fault-independent
unstable mode: the fault admittance can change by 5 orders of magnitude and
move the label by under 1%. Without this check you get a large, clean-looking,
completely uninformative dataset.

Returns `(ok, sep_deg)` and leaves `dprob.zvec` restored to `z0`.
"""
function check_self_stability(sys, dprob, z0::Vector{Float64};
                              t_final::Float64=5.0, dt::Float64=1/120,
                              eps::Float64=1e-6, tol_deg::Float64=1.0)
    gptr, _ = generator_pointers(sys)
    isempty(gptr) && return (ok=true, sep_deg=0.0)
    empty!(sys.dynamic.events)
    dprob.zvec .= z0
    dprob.zvec[gptr[1] + 4] += eps          # nudge one machine's speed
    _, traj = GradPower.integrate!(dprob, sys, t_final; dt=dt)
    sep = stability_metrics(sys, traj)["max_angle_sep_deg"]
    dprob.zvec .= z0
    return (ok = isfinite(sep) && sep <= tol_deg, sep_deg = sep)
end


# Whether this case's power flow should hold generators to their reactive
# limits. Defaults ON: a solution that leaves machines outside their nameplate
# QT/QB is not a physical operating point, and on a case the size of
# ACTIVSg2000 it is not a small error either -- 200 of 432 generators solve
# outside their limits without it, which drags 20 machines past their pull-out
# angle and makes every trajectory from that point diverge regardless of the
# fault. Set `enforce_q_limits = false` in a case table only to reproduce a
# legacy dataset.
q_limits_enabled(case) = Bool(get(case, "enforce_q_limits", true))

"""
    build_system(case) -> (sys, base)

Parse a case, form Ybus, snapshot the untouched operating point, and solve the
base power flow. Returns both the system and its `BasePoint`; see the comment
on `BasePoint` for why the snapshot has to be taken here rather than by the
caller afterwards.
"""
function build_system(case)
    raw = resolve_path(case["raw"])
    dyr = resolve_path(case["dyr"])
    sys = GradPower.from_psse(raw, dyr;
                              add_static_gen_stubs=Bool(get(case, "add_static_gen_stubs", true)),
                              surrogates=Bool(get(case, "surrogates", false)))
    GradPower.build_network!(sys)
    base = capture_base(sys)
    GradPower.runpf!(sys; verbose=false, enforce_q_limits=q_limits_enabled(case))
    return sys, base
end


# ---------------------------------------------------------------------------
# Scenario enumeration
# ---------------------------------------------------------------------------

"""
    sample_load_scales(cfg, case_idx) -> Vector{Float64}

Draw the operating-point multipliers for one case.

`lambda` is sampled from the distribution in `[load]` -- currently Uniform(low,
high). The base case is the mean when `low + high == 2`. One draw is shared by
the whole fault cross product beneath it, so the power flow is re-solved
`n_samples` times per case rather than once per simulation.

The seed is mixed with `case_idx` so different cases get different draws while
the whole sweep stays reproducible from the config alone.
"""
function sample_load_scales(cfg, case_idx::Int)
    haskey(cfg, "load") || return [1.0]
    ld = cfg["load"]
    n = Int(get(ld, "n_samples", 1))
    n >= 1 || error("[load] n_samples must be >= 1")
    dist = lowercase(String(get(ld, "distribution", "uniform")))
    seed = UInt64(get(ld, "seed", 20240917)) + UInt64(1009 * case_idx)
    rng = MersenneTwister(seed)
    if dist == "uniform"
        lo = Float64(get(ld, "low", 1.0)); hi = Float64(get(ld, "high", 1.0))
        lo <= hi || error("[load] low must be <= high")
        return [lo + (hi - lo) * rand(rng) for _ in 1:n]
    else
        error("[load] unsupported distribution \"$dist\" (only \"uniform\" so far)")
    end
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
        lambdas = sample_load_scales(cfg, ci)
        pairs = fault_bus_pairs(systems[ci], fault["buses"])
        for (li, lam) in enumerate(lambdas)
            for (internal, ext) in pairs
                for rf in r_faults, dur in durations
                    push!(scenarios,
                          Scenario(ci, lam, li - 1, internal, ext, rf, t_on, t_on + dur))
                end
            end
        end
    end
    return scenarios
end


# ---------------------------------------------------------------------------
# Channel extraction
# ---------------------------------------------------------------------------

"""
    generator_pointers(sys) -> (diff_ptr, bus_internal)

z-vector offsets of each dynamic machine's differential block, in device order.

GENROU differential layout is `[e_qp, e_dp, phi_1d, phi_2q, w, delta]`, so
`omega = z[diff_ptr + 4]` and `delta = z[diff_ptr + 5]` (see
`src/generators.jl:194-199` and `:457-462`).

We index by that offset deliberately and do NOT use `get_diff_names`:
`get_diff_names(::Genrou)` returns a differently-ordered list than the kernel
actually uses (`src/generators.jl:151`), a known discrepancy also recorded in
`study/stability_geometry/AUDIT.md`. `device.diff_ptr` is the post-cluster
reordering position, so it stays correct after `set_dynamics!`.
"""
function generator_pointers(sys)
    ptr = Int[]; bus = Int[]
    for dev in sys.dynamic.devices
        dev.dtype isa GradPower.AbstractGeneratorType || continue
        dev.dtype isa GradPower.StaticGenerator && continue   # no differential states
        push!(ptr, dev.diff_ptr)
        push!(bus, dev.dtype.bus)
    end
    return ptr, bus
end

"""
    dynamics_channels(sys, traj; downsample=1) -> Dict

Per-sample time series, all `Float32` and shaped `[n, T]`.

  * `bus_vm`, `bus_va`  -- magnitude (pu) and angle (rad) at every bus
  * `gen_delta`         -- rotor angle (rad), one row per dynamic machine
  * `gen_omega`         -- speed deviation (pu), same row order

Bus voltages live at the tail of `z` as interleaved `(vr, vi)` pairs, after the
differential and algebraic blocks.
"""
function dynamics_channels(sys, traj::AbstractMatrix; downsample::Int=1)
    downsample >= 1 || error("downsample must be >= 1")
    cols = 1:downsample:size(traj, 2)
    T = length(cols)
    nbus = length(sys.buses)
    net = sys.dynamic.diff_dim + sys.dynamic.alg_dim

    vm = Array{Float32}(undef, nbus, T)
    va = Array{Float32}(undef, nbus, T)
    @inbounds for (tj, c) in enumerate(cols), b in 1:nbus
        vr = traj[net + 2 * (b - 1) + 1, c]
        vi = traj[net + 2 * (b - 1) + 2, c]
        vm[b, tj] = hypot(vr, vi)
        va[b, tj] = atan(vi, vr)
    end

    gptr, _ = generator_pointers(sys)
    ng = length(gptr)
    delta = Array{Float32}(undef, ng, T)
    omega = Array{Float32}(undef, ng, T)
    @inbounds for (tj, c) in enumerate(cols), g in 1:ng
        omega[g, tj] = traj[gptr[g] + 4, c]
        delta[g, tj] = traj[gptr[g] + 5, c]
    end

    return Dict("bus_vm" => vm, "bus_va" => va,
                "gen_delta" => delta, "gen_omega" => omega, "T" => T)
end

"""
    dynamics_graph(sys) -> Dict

Static topology, written once per output file. Arrays become datasets and
scalars become attributes under `/grid`.

`branch_index` is 1-based `[2, n_branch]` internal bus indices (row 1 = from,
row 2 = to); `gen_bus` is the internal bus of each dynamic machine, in the same
row order as the `gen_*` channels.
"""
function dynamics_graph(sys)
    nb = length(sys.buses)
    bidx = Array{Int32}(undef, 2, length(sys.branches))
    br = Array{Float32}(undef, 4, length(sys.branches))   # r, x, sh, tap
    for (k, b) in enumerate(sys.branches)
        bidx[1, k] = b.fr; bidx[2, k] = b.to
        br[1, k] = b.r; br[2, k] = b.x; br[3, k] = b.sh; br[4, k] = b.tap
    end
    _, gbus = generator_pointers(sys)

    pd = zeros(Float32, nb); qd = zeros(Float32, nb)
    for l in sys.loads
        pd[l.bus] += l.pd; qd[l.bus] += l.qd
    end

    return Dict(
        "bus_id"       => Int32[b.i for b in sys.buses],       # external PSS/E number
        "bus_type"     => Int32[b.type for b in sys.buses],
        "bus_basekv"   => Float32[b.baseKV for b in sys.buses],
        "bus_vm0"      => Float32[b.v0m for b in sys.buses],   # base-case power flow
        "bus_va0"      => Float32[b.v0a for b in sys.buses],
        "bus_pd0"      => pd,
        "bus_qd0"      => qd,
        "branch_index" => bidx,
        "branch_param" => br,
        "gen_bus"      => Int32.(gbus),
        "n_bus"        => nb,
        "n_branch"     => length(sys.branches),
        "n_gen"        => length(gbus),
    )
end

"""
    stability_metrics(sys, traj; angle_sep_threshold, tail_fraction, decay_ratio) -> Dict

Screening labels for one trajectory.

Separation is measured RELATIVE TO THE PRE-FAULT STATE. Define the deviation
of each machine from where it started,

    dev_i(t) = delta_i(t) - delta_i(0),

and take `spread(t) = max_i dev_i(t) - min_i dev_i(t)`, which is 0 at t = 0 by
construction.

Using the ABSOLUTE spread `max_i delta_i - min_i delta_i` would be wrong for
anything larger than a toy case: a geographically large system has a large
steady-state angle spread at rest (ACTIVSg2000 sits at ~197 deg with no fault
at all), so an absolute threshold labels every sample unstable before the fault
is even applied. `initial_sep_deg` reports that standing spread for reference.

  * `max_angle_sep_deg` -- peak of `spread(t)`, i.e. how far the machines pull
    apart relative to where they started. This is a first-swing separation
    screen; it is NOT center-of-inertia referenced.
  * `max_freq_dev` -- peak `|omega|` over all machines and time (pu).
  * `tail_peak_ratio` -- peak of `spread(t)` over the final `tail_fraction` of
    the window, divided by the global peak. 0 means the swing is fully gone by
    the end; 1 means it is as large at the end as it ever was.
  * `settled` -- `tail_peak_ratio <= decay_ratio`, i.e. the swing has decayed
    to at most `decay_ratio` of its peak by the end of the window.
  * `stable` -- `max_angle_sep_deg < threshold` AND `settled` AND not diverged.

## Why `settled` is defined this way

The previous version compared the peak over the final half of the window
against the peak over the first half, and called the trajectory damped when the
tail was no larger. That is wrong for a large system, and measurably so. On a
2000-bus case the FIRST swing routinely peaks after the midpoint of a short
window, so a perfectly healthy trajectory whose swing simply has not finished
yet reads as "growing". At `t_final = 2.5` s on ACTIVSg2000 that mislabelled
90% of the unstable samples: reruns at `t_final = 10` s returned identical
separations (bus 5229: 22.50 deg either way) and flipped the label to stable.
IEEE39, whose swings settle in ~1 s, showed 0% of this error -- which is why
the bug survived until the big case was measured.

Comparing two LATE windows instead does not fix it either; it fails in the
opposite direction. Once a machine goes over the top, `spread(t)` keeps rising
to tens of thousands of degrees and can then flatten or dip, so a
late-window-vs-earlier-late-window test calls a 55,000 deg runaway "damped".
Measured on ACTIVSg2000: such a rule labelled all 24 traces stable, including
eight genuine runaways.

Both failure modes are avoided by comparing the tail against the GLOBAL peak,
which is what `tail_peak_ratio` does. Note this is only non-vacuous because the
comparison is against a fraction of the peak: `tail_peak <= peak` is trivially
true, since the tail is a subset of the window.

The `decay_ratio = 0.5` default sits in the middle of a wide measured gap
(ACTIVSg2000, 24 traces, `t_final = 10` s):

    bounded trajectories   peak    1.2 ..    45.5 deg    tail/peak  0.03 .. 0.32
    runaway trajectories   peak  38920 .. 55479   deg    tail/peak  0.72 .. 1.00

Both the separation threshold and the decay ratio separate these two
populations on their own, with orders of magnitude of margin; requiring both is
belt and braces against a bounded-but-still-growing swing that the threshold
alone would miss.

## Limits

`stable` is a coarse screen, not a certificate. It says nothing about horizons
longer than the simulated window, and near the stability boundary it is
dt-sensitive because unstable trajectories are chaotic. `t_final` must be long
enough for the first swing to resolve -- 10 s for ACTIVSg2000; 2 s is not
enough and will corrupt the labels no matter how the criterion is written.

`tail_peak_ratio` is stored on every sample so labels can be recomputed offline
under a different rule without re-simulating.
"""
function stability_metrics(sys, traj::AbstractMatrix;
                           angle_sep_threshold::Float64=deg2rad(180.0),
                           tail_fraction::Float64=0.2,
                           decay_ratio::Float64=0.5)
    0.0 < tail_fraction < 1.0 || error("tail_fraction must be in (0, 1)")
    0.0 < decay_ratio  <= 1.0 || error("decay_ratio must be in (0, 1]")
    gptr, _ = generator_pointers(sys)
    nT = size(traj, 2)
    if isempty(gptr)
        return Dict("stable" => true, "settled" => true,
                    "max_angle_sep_deg" => 0.0, "max_freq_dev" => 0.0,
                    "final_angle_sep_deg" => 0.0, "initial_sep_deg" => 0.0,
                    "tail_peak_ratio" => 0.0, "diverged" => false)
    end

    d0 = [traj[p + 5, 1] for p in gptr]
    initial_sep = maximum(d0) - minimum(d0)

    spread = Vector{Float64}(undef, nT)
    maxw = 0.0
    @inbounds for c in 1:nT
        lo = Inf; hi = -Inf
        for (i, p) in enumerate(gptr)
            dv = traj[p + 5, c] - d0[i]          # deviation from pre-fault
            dv < lo && (lo = dv); dv > hi && (hi = dv)
            w = abs(traj[p + 4, c]); w > maxw && (maxw = w)
        end
        spread[c] = hi - lo
    end

    diverged = !all(isfinite, spread) || !isfinite(maxw)
    peak = diverged ? Inf : maximum(spread)

    tail_start = max(1, nT - max(2, Int(floor(nT * tail_fraction))) + 1)
    tail_peak = diverged ? Inf : maximum(view(spread, tail_start:nT))
    # A trajectory that never moved is settled by definition; guard the divide.
    ratio = (diverged || peak <= 0.0) ? (diverged ? Inf : 0.0) : tail_peak / peak
    settled = !diverged && ratio <= decay_ratio

    return Dict(
        "stable"              => (peak < angle_sep_threshold) && settled && !diverged,
        "settled"             => settled,
        "diverged"            => diverged,
        "max_angle_sep_deg"   => rad2deg(peak),
        "final_angle_sep_deg" => rad2deg(spread[end]),
        "initial_sep_deg"     => rad2deg(initial_sep),
        "tail_peak_ratio"     => ratio,
        "max_freq_dev"        => maxw,
    )
end


# ---------------------------------------------------------------------------
# HDF5 writing
# ---------------------------------------------------------------------------

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

function write_sample!(fid, idx::Int, sc::Scenario, ch, metrics)
    grp = create_group(fid, @sprintf("samples/%06d", idx))
    write_series!(grp, "bus_vm",    ch["bus_vm"])
    write_series!(grp, "bus_va",    ch["bus_va"])
    write_series!(grp, "gen_delta", ch["gen_delta"])
    write_series!(grp, "gen_omega", ch["gen_omega"])

    a = attributes(grp)
    a["load_scale"]         = sc.load_scale
    a["load_draw"]          = sc.load_draw
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
