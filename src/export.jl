# Dynamic-stability data export.
#
# Pure functions (no IO / HDF5 dependency) that turn a solved
# `PowerSystem` + integration trajectory into graph-aligned, ML-ready
# arrays for the LUMINA `dynstab` dataset:
#
#   * `dynamics_graph(sys)`      -> static topology (nodes + edges), the
#                                   graph each sample lives on.
#   * `dynamics_channels(sys, traj; downsample)` -> per-node time series
#                                   (bus vm/va, generator rotor angle/speed).
#   * `stability_metrics(sys, traj; ...)` -> scalar transient-stability
#                                   summary + a default stable/unstable label.
#
# The HDF5 writer lives in `scripts/dynstab_io.jl`, keeping the core
# package free of an HDF5 dependency. Both sides agree on EXPORT_SCHEMA_VERSION.
#
# ---------------------------------------------------------------------------
# State-layout note (load-bearing):
#   The trajectory `z = [diff_states; alg_states; v_re/v_im per bus]`.
#   GENROU's diff block is, per the batched kernel (src/kernels/genrou.jl):
#       diff_ptr+0 e_qp, +1 e_dp, +2 phi_1d, +3 phi_2q, +4 omega, +5 delta
#   This is the AUTHORITATIVE ordering (it matches `gen_speeds`, `coupling.jl`
#   `w_offset`, and the README). NOTE: `get_diff_names(::Genrou)` returns a
#   STALE/incorrect order and must NOT be used here.
# ---------------------------------------------------------------------------

# Schema history:
#   1 — bus_vm/bus_va/gen_delta/gen_omega channels; `stable` thresholded on the
#       raw inter-machine spread (baseline-contaminated, see stability_metrics).
#   2 — adds prefault/final spread, COI-referenced excursion (max + final) and
#       a `finite` flag; `stable` now thresholds the COI excursion. Channels
#       and /grid are UNCHANGED. `max_angle_sep_rad|deg`, `max_freq_dev`,
#       `settle_freq_dev`, `damped` keep their v1 meaning and values.
#   3 — both loss-of-synchronism labels are exported side by side:
#       `stable_coi` (the v2 criterion) and `stable_spread` (the v1 criterion,
#       now also requiring a finite trajectory). `stable` is kept as an alias
#       of `stable_coi`. Operating-point provenance (`op_id` per sample,
#       `/operating_points/<id>` per file) is written by the driver.
const EXPORT_SCHEMA_VERSION = 3

# zero-based diff offsets within a GENROU machine's diff block.
const _GENROU_OMEGA_OFF = 4
const _GENROU_DELTA_OFF = 5

"""
    _genrou_nodes(sys) -> Vector{NamedTuple}

Generator dynamic nodes carrying rotor states, in device order. Each entry is
`(bus_internal, bus_ext, diff_ptr, mbase, H)`. Only `Genrou` machines are dynamic
nodes; `Gensal` is a placeholder with no states, and `StaticGenerator` stubs
have no rotor. `dynamics_graph` and `dynamics_channels` MUST iterate this same
list so the generator node ordering matches the channel rows.
"""
function _genrou_nodes(sys::PowerSystem)
    nodes = NamedTuple{(:bus_internal, :bus_ext, :diff_ptr, :mbase, :H),
                       Tuple{Int,Int,Int,Float64,Float64}}[]
    dyn = sys.dynamic
    dyn === nothing && return nodes
    for (i, device) in enumerate(dyn.devices)
        device.dtype isa Genrou || continue
        bus_internal = dyn.map.bus[i]
        gidx = dyn.map.gen[i]
        mbase = gidx > 0 ? sys.gens[gidx].mbase : NaN
        # `set_dynamics!` has already rescaled H by mbase/baseMVA, so this is
        # inertia on the SYSTEM base — the correct weight for a COI reference.
        H = device.dtype.H
        push!(nodes, (bus_internal = bus_internal,
                      bus_ext = sys.buses[bus_internal].i,
                      diff_ptr = device.diff_ptr,
                      mbase = mbase,
                      H = H))
    end
    return nodes
end

"""
    dynamics_graph(sys::PowerSystem) -> Dict{String,Any}

Static topology of the system as graph-aligned arrays. Bus channels and
`branch_fr`/`branch_to` use INTERNAL 1-based bus indices (the ordering of
`sys.buses`); `*_ext` arrays carry the external PSS/E bus numbers. The Python
packaging step decides ac_line vs transformer edge types from
`branch_tap`/`branch_shift`.
"""
function dynamics_graph(sys::PowerSystem)
    nbus = length(sys.buses)
    gnodes = _genrou_nodes(sys)

    g = Dict{String,Any}()
    # --- bus nodes ---
    g["bus_i"]       = Int32[b.i for b in sys.buses]          # external number
    g["bus_type"]    = Int32[b.type for b in sys.buses]
    g["bus_base_kv"] = Float32[b.baseKV for b in sys.buses]
    g["bus_v0m"]     = Float32[b.v0m for b in sys.buses]      # PF voltage mag
    g["bus_v0a"]     = Float32[b.v0a for b in sys.buses]      # PF voltage angle

    # --- generator (dynamic machine) nodes ---
    g["gen_bus"]     = Int32[n.bus_internal for n in gnodes]  # internal 1-based
    g["gen_bus_ext"] = Int32[n.bus_ext for n in gnodes]
    g["gen_mbase"]   = Float32[n.mbase for n in gnodes]

    # --- branches (edges) ---
    g["branch_fr"]    = Int32[br.fr for br in sys.branches]   # internal 1-based
    g["branch_to"]    = Int32[br.to for br in sys.branches]
    g["branch_r"]     = Float32[br.r for br in sys.branches]
    g["branch_x"]     = Float32[br.x for br in sys.branches]
    g["branch_sh"]    = Float32[br.sh for br in sys.branches]
    g["branch_tap"]   = Float32[br.tap for br in sys.branches]
    g["branch_shift"] = Float32[br.shift for br in sys.branches]

    # --- loads ---
    g["load_bus"] = Int32[ld.bus for ld in sys.loads]        # internal 1-based
    g["load_pd"]  = Float32[ld.pd for ld in sys.loads]
    g["load_qd"]  = Float32[ld.qd for ld in sys.loads]

    # --- shunts ---
    g["shunt_bus"] = Int32[sh.bus for sh in sys.shunts]      # internal 1-based
    g["shunt_gsh"] = Float32[sh.gsh for sh in sys.shunts]
    g["shunt_bsh"] = Float32[sh.bsh for sh in sys.shunts]

    g["n_bus"]    = nbus
    g["n_gen"]    = length(gnodes)
    g["n_branch"] = length(sys.branches)
    g["n_load"]   = length(sys.loads)
    g["n_shunt"]  = length(sys.shunts)
    g["baseMVA"]  = sys.baseMVA
    return g
end

"""
    dynamics_channels(sys, traj; downsample=1) -> Dict{String,Any}

Per-node time series extracted from the integration trajectory `traj`
(`system_size × nsteps`). Returns Float32 matrices:

  - `bus_vm`, `bus_va`      : `(n_bus × T)`  voltage magnitude / angle (rad)
  - `gen_delta`, `gen_omega`: `(n_gen × T)`  rotor angle (rad) / speed deviation (pu)

`downsample = k` keeps every k-th time step (`T = ceil(nsteps/k)`); the first
and last samples are always retained. Generator rows follow `_genrou_nodes`
order (same as `dynamics_graph`'s `gen_*`).
"""
function dynamics_channels(sys::PowerSystem, traj::AbstractMatrix; downsample::Int=1)
    @assert downsample >= 1 "downsample must be >= 1"
    nbus = length(sys.buses)
    diff_dim = sys.dynamic.diff_dim
    alg_dim = sys.dynamic.alg_dim
    voff = diff_dim + alg_dim
    nsteps = size(traj, 2)

    # time index selection: every k-th, always include the last step.
    idx = collect(1:downsample:nsteps)
    if idx[end] != nsteps
        push!(idx, nsteps)
    end
    T = length(idx)

    # --- bus voltage channels (v_re, v_im) -> vm, va ---
    bus_vm = Array{Float32}(undef, nbus, T)
    bus_va = Array{Float32}(undef, nbus, T)
    @inbounds for (tj, t) in enumerate(idx)
        for b in 1:nbus
            vre = traj[voff + 2*(b-1) + 1, t]
            vim = traj[voff + 2*(b-1) + 2, t]
            bus_vm[b, tj] = sqrt(vre*vre + vim*vim)
            bus_va[b, tj] = atan(vim, vre)
        end
    end

    # --- generator rotor channels ---
    gnodes = _genrou_nodes(sys)
    ng = length(gnodes)
    gen_delta = Array{Float32}(undef, ng, T)
    gen_omega = Array{Float32}(undef, ng, T)
    @inbounds for (gj, n) in enumerate(gnodes)
        wrow = n.diff_ptr + _GENROU_OMEGA_OFF
        drow = n.diff_ptr + _GENROU_DELTA_OFF
        for (tj, t) in enumerate(idx)
            gen_omega[gj, tj] = traj[wrow, t]
            gen_delta[gj, tj] = traj[drow, t]
        end
    end

    return Dict{String,Any}(
        "bus_vm"    => bus_vm,
        "bus_va"    => bus_va,
        "gen_delta" => gen_delta,
        "gen_omega" => gen_omega,
        "time_idx"  => Int32.(idx),
        "T"         => T,
    )
end

"""
    stability_metrics(sys, traj; angle_sep_threshold=pi, settle_ratio=0.5,
                      settle_frac=0.1) -> Dict{String,Any}

Transient-stability summary computed from the FULL-resolution trajectory
(rotor channels).

## Why the excursion, not the raw spread

The raw inter-machine spread `max_i delta_i - min_i delta_i` is NOT a usable
label on its own: it includes the PRE-FAULT steady-state spread, which is a
property of the dispatch, not of the disturbance. On IEEE39 that baseline is
~50 deg, so a fixed 180 deg threshold on the raw spread leaves only ~130 deg of
headroom and lands inside the natural first-swing range — the label then flips
on numerical noise for every mild fault. IEEE9's baseline is ~29 deg, so the
same threshold means something different on each case.

The default label therefore uses the **COI-referenced excursion from the
pre-fault equilibrium**: with an inertia-weighted centre of angle
`delta_coi(t) = sum(H_i delta_i) / sum(H_i)` and `dt_i(t) = delta_i(t) - delta_coi(t)`,

    excursion(t) = max_i | dt_i(t) - dt_i(1) |

This is zero at `t = 1` for every case, measures only what the disturbance did,
and diverges exactly when a machine pulls away from the pack. Loss of
synchronism is `max_t excursion(t) >= angle_sep_threshold`.

## Returned metrics

Baseline / excursion (the label basis):
  - `prefault_angle_sep_rad|deg` : inter-machine spread at `t = 1`.
  - `max_coi_excursion_rad|deg`  : peak COI-referenced excursion (above).
  - `final_coi_excursion_rad|deg`: excursion at the last step — separates a
                                   bounded swing from a run-away.

Raw spread (kept for backward compatibility and re-derivation):
  - `max_angle_sep_rad|deg`   : peak raw inter-machine spread.
  - `final_angle_sep_rad|deg` : raw spread at the last step.

Speed:
  - `max_freq_dev`    : peak `|omega|` (pu speed deviation) over all machines.
  - `settle_freq_dev` : peak `|omega|` over the trailing `settle_frac`.

Labels (both criteria are exported; which is canonical is still open):
  - `stable_coi`    : `max_coi_excursion_rad < angle_sep_threshold` AND the
                      trajectory is finite.
  - `stable_spread` : `max_angle_sep_rad < angle_sep_threshold` AND the
                      trajectory is finite — the v1 raw-spread criterion. It
                      includes the pre-fault spread, so its headroom depends
                      on the dispatch (IEEE39 ~50 deg, ACTIVSg2000 ~101 deg).
  - `stable`        : alias of `stable_coi`, kept for existing consumers.
  - `damped`   : trailing-window speed deviation decayed below
                 `settle_ratio * max_freq_dev`. Reported SEPARATELY and never
                 folded into `stable` — a bounded but lightly-damped swing is
                 still transiently stable on a finite horizon.

                 Two caveats. It is a RATIO, so it is only meaningful when
                 `stable` is true: a run-away case can show `damped == true`
                 simply because `max_freq_dev` blew up faster than the tail.
                 And it is HORIZON-DEPENDENT — it asks "did this settle within
                 `t_final`", not "is this damped". On IEEE39 (D = 0, no AVR)
                 only ~12% of stable samples are `damped` at `t_final = 10 s`,
                 while the same faults do settle by 20 s. Filter on
                 `stable && damped`, and keep `t_final` fixed across a dataset.
  - `finite`   : no NaN/Inf in the rotor channels. A diverged integration is
                 labelled unstable rather than silently producing NaN metrics.

All raw metrics are returned so labels can be re-derived downstream in Python.

Single-machine systems (machine vs. infinite bus) have no inter-machine spread;
both the spread and the excursion fall back to `|delta(t) - delta(1)|`, which
is the physically meaningful quantity there.

!!! note "Requires a true equilibrium at t = 1"
    Every metric is referenced to step 1, so the initial condition must be an
    actual steady state. Verify with `rhs_fun!(f, z0, u, p, sys)`;
    `norm(f, Inf)` should be ~1e-10 or smaller. If the pre-fault point drifts,
    these labels describe the drift, not the fault.
"""
function stability_metrics(sys::PowerSystem, traj::AbstractMatrix;
                           angle_sep_threshold::Float64=Float64(pi),
                           settle_ratio::Float64=0.5,
                           settle_frac::Float64=0.1)
    gnodes = _genrou_nodes(sys)
    ng = length(gnodes)
    nsteps = size(traj, 2)

    if ng == 0
        return Dict{String,Any}(
            "n_gen" => 0,
            "prefault_angle_sep_rad" => NaN, "prefault_angle_sep_deg" => NaN,
            "max_angle_sep_rad" => NaN, "max_angle_sep_deg" => NaN,
            "final_angle_sep_rad" => NaN, "final_angle_sep_deg" => NaN,
            "max_coi_excursion_rad" => NaN, "max_coi_excursion_deg" => NaN,
            "final_coi_excursion_rad" => NaN, "final_coi_excursion_deg" => NaN,
            "max_freq_dev" => NaN, "settle_freq_dev" => NaN,
            "stable" => true, "stable_coi" => true, "stable_spread" => true,
            "damped" => true, "finite" => true,
            "angle_sep_threshold_rad" => angle_sep_threshold,
            "settle_ratio" => settle_ratio,
        )
    end

    drows = Int[n.diff_ptr + _GENROU_DELTA_OFF for n in gnodes]
    wrows = Int[n.diff_ptr + _GENROU_OMEGA_OFF for n in gnodes]

    # Inertia weights for the COI. Fall back to a uniform mean if H is missing
    # or non-positive for any machine, so the reference is always well defined.
    Hw = Float64[n.H for n in gnodes]
    if !all(h -> isfinite(h) && h > 0, Hw)
        fill!(Hw, 1.0)
    end
    Hsum = sum(Hw)

    @inline function _coi(t)
        s = 0.0
        @inbounds for g in 1:ng
            s += Hw[g] * traj[drows[g], t]
        end
        return s / Hsum
    end

    # Pre-fault COI-referenced position of each machine (the reference state).
    coi1 = _coi(1)
    rel1 = Vector{Float64}(undef, ng)
    @inbounds for g in 1:ng
        rel1[g] = traj[drows[g], 1] - coi1
    end

    finite = true
    max_sep = 0.0
    prefault_sep = 0.0
    final_sep = 0.0
    max_exc = 0.0
    final_exc = 0.0
    max_freq = 0.0
    settle_freq = 0.0
    settle_start = max(1, Int(floor(nsteps * (1.0 - settle_frac))))

    @inbounds for t in 1:nsteps
        # NaN/Inf must be detected EXPLICITLY: `NaN < x` and `NaN > x` are both
        # false, so a non-finite rotor angle would otherwise be silently skipped
        # by the min/max reductions and a diverged run would look stable.
        for g in 1:ng
            isfinite(traj[drows[g], t]) || (finite = false)
        end

        coit = _coi(t)

        if ng >= 2
            dmin = Inf; dmax = -Inf
            for g in 1:ng
                d = traj[drows[g], t]
                d < dmin && (dmin = d)
                d > dmax && (dmax = d)
            end
            sep = dmax - dmin
        else
            sep = abs(traj[drows[1], t] - traj[drows[1], 1])
        end

        # COI-referenced excursion from the pre-fault equilibrium.
        exc = 0.0
        if ng >= 2
            for g in 1:ng
                e = abs((traj[drows[g], t] - coit) - rel1[g])
                e > exc && (exc = e)
            end
        else
            exc = abs(traj[drows[1], t] - traj[drows[1], 1])
        end

        isfinite(sep) && isfinite(exc) || (finite = false)
        sep > max_sep && (max_sep = sep)
        exc > max_exc && (max_exc = exc)
        t == 1 && (prefault_sep = sep)
        t == nsteps && (final_sep = sep; final_exc = exc)

        for g in 1:ng
            w = abs(traj[wrows[g], t])
            isfinite(w) || (finite = false)
            w > max_freq && (max_freq = w)
            if t >= settle_start && w > settle_freq
                settle_freq = w
            end
        end
    end

    # Loss of synchronism, measured as a departure from the pre-fault
    # equilibrium rather than an absolute spread. Settling is reported
    # separately as `damped` (NOT folded into `stable`).
    stable = finite && max_exc < angle_sep_threshold
    stable_spread = finite && max_sep < angle_sep_threshold
    damped = max_freq <= 0 ? true : (settle_freq <= settle_ratio * max_freq)
    return Dict{String,Any}(
        "n_gen" => ng,
        "prefault_angle_sep_rad" => prefault_sep,
        "prefault_angle_sep_deg" => rad2deg(prefault_sep),
        "max_angle_sep_rad" => max_sep,
        "max_angle_sep_deg" => rad2deg(max_sep),
        "final_angle_sep_rad" => final_sep,
        "final_angle_sep_deg" => rad2deg(final_sep),
        "max_coi_excursion_rad" => max_exc,
        "max_coi_excursion_deg" => rad2deg(max_exc),
        "final_coi_excursion_rad" => final_exc,
        "final_coi_excursion_deg" => rad2deg(final_exc),
        "max_freq_dev" => max_freq,
        "settle_freq_dev" => settle_freq,
        "stable" => stable,
        "stable_coi" => stable,
        "stable_spread" => stable_spread,
        "damped" => damped,
        "finite" => finite,
        "angle_sep_threshold_rad" => angle_sep_threshold,
        "settle_ratio" => settle_ratio,
    )
end
