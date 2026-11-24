# PERC2 core: pure-Julia, time-stepped model of one PERC2 block.
#
# Partitioned formulation: the network sees a constant current
# `(Ip - j Iq) V/|V|` (system-base pu) held during each Newton solve;
# `perc2_step!` advances filters, limits, the cease/reconnect logic and the
# Tt output lag once per ACCEPTED step. It must never be called from residual
# or Jacobian evaluations.
#
# Step order: filters -> cease logic -> current command -> limits -> FracOn ->
# Tt lag. The call at time t_k with the converged bus voltage of t_k gives the
# output at t_k; that output is the current held during step k -> k+1.
#
# Model behaviour, per the published PERC2 specification and block diagram:
# - The bus voltage is floored at 1e-3 pu before every use, so a 0 pu input
#   gives finite currents. The frequency gate (input held at nominal below
#   0.7 pu) uses this floored *raw* bus voltage, not Vtf.
# - Long point 3 (VLC3/VHC3) is timed by its own TVLC3/TVHC3.
# - Tt is floored at `tt_floor_steps*dt` (default 2 steps).
# - Reactive path: Mvarinit = QPratio·MWinit (QPratio is a ratio to the
#   initial active POWER P0, not to Mbase; consumed Mvar positive). Steady
#   state Q = Mvarinit (Vtf/V0)^nQ, i.e. Iqv = (Mvarinit/Mbase)(Vtf/V0)^nQ / Vtf,
#   limited to [IqminUsed, IqmaxUsed] with IqmaxUsed = min(Iqmax, sqrt(Imax² - Ip²))
#   and IqminUsed = max(Iqmin, -sqrt(Imax² - Ip²)), where Ip is the limited
#   active current command (active-current priority). FracOn and the Tt lag
#   apply to both paths.
# - Iqextra: the mismatch between the power-flow Q0 and Mvarinit is a
#   CONSTANT CURRENT
#     Iqextra = (Q0 − Mvarinit) / (Mbase·V0)            (model base)
#   added AFTER the Iq limiter and BEFORE the FracOn multiplier:
#     Iq_out = Tt-lag( FracOn · (Iqv + Iqextra) ).
#   So Iqextra bypasses nQ, the Q washout and lead-lag, and Iqmin/Iqmax/Imax,
#   but it ceases and reconnects with FracOn and passes through Tt.
#   `use_qpratio = false` takes the stand-alone reading Mvarinit = Q0
#   (Iqextra = 0, all of Q0 on the dynamic Q path). When |Q0 − QPratio·P0| is
#   within rounding (8 eps), Mvarinit is set to Q0 exactly, so QPratio = Q0/P0
#   gives results bit-identical to `use_qpratio = false`.
# - Ldfact ≤ 0 falls back to 0.8; 0 < Ldfact is used as given.
#
# Transient blocks (washouts and lead-lags):
# - Topology:
#     Pctrl = LL_P( MWinit/MBase + Kdroop·(ftf − edge) + W_P(Vterm) )
#     Qctrl = LL_Q( Mvarinit/MBase + W_Q(Vterm) )
#   then Ip = Pctrl·(Vtf/Vinit)^nP / Vtf, Iq = Qctrl·(Vtf/Vinit)^nQ / Vtf,
#   limits, FracOn, Tt. MWinit/MBase = Ldfact.
# - Washout input: the raw bus voltage Vterm (floored at 1e-3), not Vtf.
# - Washout sign and units: W(s) = sK/(1+sT), added with a + sign, on the
#   model base. A voltage drop lowers the load and a rise raises it. Its step
#   response is (K/T)·ΔV·e^{−t/T}.
#   Realization: x' = (u − x)/T, y = (K/T)(u − x).
#   Initialization: x = Vinit, so y = 0.
# - Lead-lag (1+sTa)/(1+sTb) on the summed request, before the exponent.
#   Realization: x' = (u − x)/Tb, y = x + (Ta/Tb)(u − x). Initialization:
#   x = u0 (Ldfact, or Q0/MBase), so y = u0. Tb ≤ 0 bypasses the block and
#   ignores Ta; validate() rejects Ta > 0 with Tb = 0. Ta == Tb is also
#   bypassed (y = u exactly).
# - Discretization: the input sampled at t_k is held over the step, and the
#   state is advanced with the exact factor 1 − exp(−dt/T) before the output
#   is formed, as for the Tv/Tf/Tt filters.
# - Disabled blocks (K == 0; Tb ≤ 0 or Ta == Tb) do no arithmetic on the
#   request, so results are bit-identical to a model without them.
# - The washout and lead-lag act before the limits, so their transients are
#   clipped by Ipmax/Iqmax/Imax like the rest of the request.

const PERC2_MIN_VOLTAGE = 1e-3
const PERC2_FREQ_GATE_PU = 0.7
const PERC2_MONITOR, PERC2_CEASED, PERC2_RAMP = 0, 1, 2
const _PERC2_EPS = 1e-9

# One cease check: level 0 momentary, 1 long, 2 frequency.
struct PERC2Check
    name::Symbol
    level::Int
    low::Bool               # true: x < threshold; false: x > threshold
    freq::Bool              # true: compares frequency (Hz), else Vtf (pu)
    threshold::Float64
    delay::Float64
end

_perc2_active(c::PERC2Check, vtf, f_hz) =
    (x = c.freq ? f_hz : vtf; c.low ? x < c.threshold : x > c.threshold)

function _perc2_build_checks(p::PERC2Params)
    checks = PERC2Check[]
    p.VLMflag && push!(checks, PERC2Check(:VLM, 0, true, false, p.VLcease, p.Tcease))
    p.VHMflag && push!(checks, PERC2Check(:VHM, 0, false, false, p.VHcease, p.Tcease))
    if p.VLLflag
        push!(checks, PERC2Check(:VLC1, 1, true, false, p.VLC1, p.TVLC1),
                      PERC2Check(:VLC2, 1, true, false, p.VLC2, p.TVLC2),
                      PERC2Check(:VLC3, 1, true, false, p.VLC3, p.TVLC3),
                      PERC2Check(:VLC4, 1, true, false, p.VLC4, p.TVLC4))
    end
    if p.VLHflag
        push!(checks, PERC2Check(:VHC1, 1, false, false, p.VHC1, p.TVHC1),
                      PERC2Check(:VHC2, 1, false, false, p.VHC2, p.TVHC2),
                      PERC2Check(:VHC3, 1, false, false, p.VHC3, p.TVHC3),
                      PERC2Check(:VHC4, 1, false, false, p.VHC4, p.TVHC4))
    end
    p.FLflag && push!(checks, PERC2Check(:FLC1, 2, true, true, p.FLC1, p.TFLC1))
    p.FHflag && push!(checks, PERC2Check(:FHC1, 2, false, true, p.FHC1, p.TFHC1))
    return checks
end

mutable struct PERC2State
    # operating point (system base)
    P0::Float64
    Q0::Float64
    V0::Float64
    mbase::Float64          # model MVA base, system pu: P0 / Ldfact
    # held output currents (model base, after Tt lag)
    ip_m::Float64
    iq_m::Float64
    # measurements
    vtf::Float64
    ftf::Float64            # filtered frequency deviation, pu of fnom
    # logic
    frac::Float64
    tt_floor_applied::Bool
    # derived constants
    ldfact::Float64
    q0_m::Float64           # Mvarinit on model base (dynamic Q path)
    iqx_m::Float64          # Iqextra, model base (constant current, scaled by FracOn)
    fcease::Float64         # clamped to [0, 1]
    floor::Float64          # 1 - fcease
    final::Float64          # (1 - fcease) + max(frecon, 0) fcease
    vlrecon::Float64        # max(VLrecon, VLcease)
    vhrecon::Float64        # min(VHrecon, VHcease)
    checks::Vector{PERC2Check}
    # cease/reconnect machine
    timers::Vector{Float64} # NaN = not running
    mode::Int
    level::Int              # -1 none, 0 momentary, 1 long, 2 frequency
    pending::Int            # index into checks waiting out Tdelay; 0 = none
    pending_t::Float64
    recon_t0::Float64       # NaN = not in the reconnect window
    ramp_t0::Float64
    ramp_from::Float64
    trecon_used::Float64
    tramp_used::Float64
    cause::Symbol
    first_step::Bool
    # transient blocks (model base): washout lag states (track the raw bus
    # voltage) and lead-lag states (track the lead-lag inputs)
    xwp::Float64
    xwq::Float64
    xllp::Float64
    xllq::Float64
    events::Vector{Tuple{Symbol,Float64,Symbol}}  # (:trip | :cease | :reconnect_start | :reconnect_end, t, check)
end

_perc2_ldfact(p::PERC2Params) = p.Ldfact > 0 ? p.Ldfact : 0.8

# Mvarinit (system pu): QPratio·P0, snapped to Q0 within rounding; Q0 when
# `use_qpratio == false`.
function _perc2_mvarinit(p::PERC2Params, P0, Q0)
    p.use_qpratio || return Float64(Q0)
    m = p.QPratio * P0
    return abs(Q0 - m) <= 8 * eps(max(abs(Q0), abs(m), abs(P0))) ? Float64(Q0) : m
end

# (IpmaxUsed, IpminUsed) = (min(Ipmax, |Imax|), max(Ipmin, -|Imax|)), per the
# PERC2 block diagram.
function _perc2_ip_limits(p::PERC2Params)
    imax = abs(p.Imax)
    return min(p.Ipmax, imax), max(p.Ipmin, -imax)
end

# (IqmaxUsed, IqminUsed) for active-current order `ip` (active priority).
function _perc2_iq_limits(p::PERC2Params, ip)
    imax = abs(p.Imax)
    room = sqrt(max(imax * imax - ip * ip, 0.0))
    return min(p.Iqmax, room), max(p.Iqmin, -room)
end

# Droop term Kdroop (ftf - edge) on the model base.
function _perc2_droop(p::PERC2Params, ftf)
    edge = ftf < p.dbfl ? p.dbfl : min(ftf, p.dbfh)
    return p.Kdroop * (ftf - edge)
end

# Limited (Ip, Iq) command on the model base before FracOn, from the power
# requests `pctrl`, `qctrl` (block-diagram states 1 and 2, model base, defined
# at Vinit).
function _perc2_command(s::PERC2State, p::PERC2Params, vtf, pctrl, qctrl)
    vden = max(vtf, PERC2_MIN_VOLTAGE)
    ipmax, ipmin = _perc2_ip_limits(p)
    ip = _perc2_clamp(pctrl * (vtf / s.V0)^p.nP / vden, ipmin, ipmax)
    iqmax, iqmin = _perc2_iq_limits(p, ip)
    iq = _perc2_clamp(qctrl * (vtf / s.V0)^p.nQ / vden, iqmin, iqmax)
    return ip, iq
end

# Washout sK/(1+sT): x' = (u - x)/T, y = (K/T)(u - x). Disabled (y = 0, state
# untouched) when K == 0. Advances x with the exact exponential update.
@inline function _perc2_washout!(x, u, K, T, dt)
    K == 0 && return x, 0.0
    x += (u - x) * _perc2_alpha(T, dt)
    return x, (K / T) * (u - x)
end

# Lead-lag (1+sTa)/(1+sTb): x' = (u - x)/Tb, y = x + (Ta/Tb)(u - x). Bypassed
# (y = u exactly) when Tb <= 0 (Ta then ignored) or Ta == Tb.
@inline _perc2_ll_bypass(Ta, Tb) = Tb <= 0 || Ta == Tb
@inline function _perc2_leadlag!(x, u, Ta, Tb, dt)
    _perc2_ll_bypass(Ta, Tb) && return x, u
    x += (u - x) * _perc2_alpha(Tb, dt)
    return x, x + (Ta / Tb) * (u - x)
end

# min(max(x, lo), hi) (hi wins when lo > hi).
_perc2_clamp(x, lo, hi) = min(max(x, lo), hi)

_perc2_alpha(T, dt) = T <= 0 ? 1.0 : -expm1(-dt / T)

"""
    perc2_init(p, P0, Q0, V0) -> PERC2State

Steady state for consumed power `P0 > 0`, `Q0` (system-base pu, the power-flow
values) at bus voltage `V0 > 0` (pu). The model base is `Mbase = P0 / Ldfact`
(`Ldfact ≤ 0` → 0.8). The dynamic reactive path starts at
`Mvarinit = QPratio·P0`; the rest, `Q0 − Mvarinit`, is drawn as the constant
current `Iqextra = (Q0 − Mvarinit)/V0` (system base) that follows FracOn and Tt
but bypasses nQ, the Q washout/lead-lag and the current limits.
`use_qpratio = false` sets `Mvarinit = Q0`. `perc2_output` returns the total
Iq including Iqextra.
Calls `validate(p)`. If the current limits bind at `V0` the block starts at
the limited current, so the drawn power differs from `(P0, Q0)`.
"""
function perc2_init(p::PERC2Params, P0::Real, Q0::Real, V0::Real)
    validate(p)
    P0 > 0 || throw(ArgumentError("PERC2 needs a positive initial active power (got P0 = $P0)"))
    V0 > 0 || throw(ArgumentError("PERC2 needs a positive initial voltage (got V0 = $V0)"))
    ldfact = _perc2_ldfact(p)
    mbase = P0 / ldfact
    v0 = max(Float64(V0), PERC2_MIN_VOLTAGE)
    mvarinit = _perc2_mvarinit(p, P0, Q0)
    q0_m = mvarinit / mbase
    iqx_m = mvarinit == Q0 ? 0.0 : (Q0 - mvarinit) / (mbase * v0)
    fcease = clamp(p.fcease, 0.0, 1.0)
    checks = fcease > 0 ? _perc2_build_checks(p) : PERC2Check[]
    s = PERC2State(P0, Q0, v0, mbase, 0.0, 0.0, v0, 0.0, 1.0, false,
                   ldfact, q0_m, iqx_m, fcease, 1.0 - fcease, (1.0 - fcease) + max(p.frecon, 0.0) * fcease,
                   max(p.VLrecon, p.VLcease), min(p.VHrecon, p.VHcease), checks,
                   fill(NaN, length(checks)), PERC2_MONITOR, -1, 0, NaN, NaN, NaN, 1.0,
                   max(p.Trecon, 0.0), max(p.Tramp, 0.0) * ldfact, :none, true,
                   v0, v0, ldfact, q0_m,
                   Tuple{Symbol,Float64,Symbol}[])
    s.ip_m, iqv = _perc2_command(s, p, v0, ldfact, q0_m)
    s.iq_m = iqv + iqx_m
    return s
end

"""
    perc2_output(s) -> (Ip, Iq)

System-base active and reactive current magnitudes drawn by the load, held
constant during the next Newton solve. The injected current into the network
is `-(Ip - j*Iq) * V/|V|`.
"""
perc2_output(s::PERC2State) = (s.ip_m * s.mbase, s.iq_m * s.mbase)

function _perc2_used(s::PERC2State, p::PERC2Params, c::PERC2Check)
    c.level == 0 && return max(p.Trecon, 0.0), max(p.Tramp, 0.0) * s.ldfact
    c.level == 1 && c.low && return max(p.TLreconL, 0.0), max(p.TLrampL, 0.0) * s.ldfact
    c.level == 1 && return max(p.TLreconH, 0.0), max(p.TLrampH, 0.0) * s.ldfact
    return Inf, Inf
end

function _perc2_cease!(s::PERC2State, p::PERC2Params, i::Int, t)
    c = s.checks[i]
    if s.mode == PERC2_CEASED          # escalation: already off, only the reconnect settings change
        s.level = c.level
    else
        s.mode, s.level, s.frac = PERC2_CEASED, c.level, s.floor
        push!(s.events, (:trip, t, c.name))
    end
    s.cause = c.name
    s.trecon_used, s.tramp_used = _perc2_used(s, p, c)
    s.recon_t0 = NaN
    s.pending = 0
    push!(s.events, (:cease, t, c.name))
    return s
end

# Cease/reconnect state machine: cease checks with their timers, Tdelay,
# escalation to a higher level while ceased, the strict reconnect window
# (VLrecon < Vtf < VHrecon) held for Trecon, and the ramp. A new cease
# interrupts a ramp.
function _perc2_logic!(s::PERC2State, p::PERC2Params, t, vtf, f_hz)
    eps = _PERC2_EPS
    for (i, c) in enumerate(s.checks)
        if !(s.mode != PERC2_CEASED || c.level > s.level)
            s.timers[i] = NaN
            continue
        end
        if _perc2_active(c, vtf, f_hz)
            isnan(s.timers[i]) && (s.timers[i] = t)
            if t - s.timers[i] >= max(c.delay, 0.0) - eps &&
               (s.pending == 0 || c.level > s.checks[s.pending].level)
                s.pending, s.pending_t = i, t
            end
        else
            s.timers[i] = NaN
        end
    end
    if s.pending != 0 && t - s.pending_t >= max(p.Tdelay, 0.0) - eps
        i = s.pending
        if s.mode != PERC2_CEASED || s.checks[i].level > s.level
            _perc2_cease!(s, p, i, t)
        end
        s.pending = 0
    end
    if s.mode == PERC2_CEASED && (s.level == 0 || s.level == 1)
        if s.vlrecon < vtf < s.vhrecon
            isnan(s.recon_t0) && (s.recon_t0 = t)
            if t - s.recon_t0 >= s.trecon_used - eps
                s.mode, s.ramp_t0, s.ramp_from = PERC2_RAMP, t, s.frac
                fill!(s.timers, NaN)   # every check restarts with the ramp
                push!(s.events, (:reconnect_start, t, s.cause))
            end
        else
            s.recon_t0 = NaN
        end
    end
    if s.mode == PERC2_RAMP
        progress = s.tramp_used <= 0 ? 1.0 : min((t - s.ramp_t0) / s.tramp_used, 1.0)
        s.frac = s.ramp_from + (s.final - s.ramp_from) * progress
        if progress >= 1.0
            s.mode, s.level, s.frac = PERC2_MONITOR, -1, s.final
            push!(s.events, (:reconnect_end, t, s.cause))
        end
    end
    return s.frac
end

"""
    perc2_step!(s, p, vm, f_hz, t, dt) -> (Ip, Iq)

Advance one accepted step using the converged bus voltage magnitude `vm`
(pu), the local frequency estimate `f_hz` and the time `t` of that converged
solution. Returns the new held currents (system base). At the operating point
(`vm == V0`, `f_hz == fnom`, no event) the state is unchanged.
"""
function perc2_step!(s::PERC2State, p::PERC2Params, vm::Real, f_hz::Real, t::Real, dt::Real)
    v = max(Float64(vm), PERC2_MIN_VOLTAGE)
    # filters
    f_dev = v >= PERC2_FREQ_GATE_PU ? f_hz / p.fnom - 1.0 : 0.0
    s.vtf += (v - s.vtf) * _perc2_alpha(p.Tv, dt)
    s.ftf += (f_dev - s.ftf) * _perc2_alpha(p.Tf, dt)
    # cease / reconnect logic
    frac = isempty(s.checks) ? s.frac : _perc2_logic!(s, p, Float64(t), s.vtf, p.fnom * (1 + s.ftf))
    # washouts on the raw bus voltage, droop, lead-lags -> power requests
    s.xwp, wp = _perc2_washout!(s.xwp, v, p.Kvp, p.Tvp, dt)
    s.xwq, wq = _perc2_washout!(s.xwq, v, p.Kvq, p.Tvq, dt)
    up = s.ldfact + _perc2_droop(p, s.ftf)
    p.Kvp == 0 || (up += wp)
    uq = p.Kvq == 0 ? s.q0_m : s.q0_m + wq
    s.xllp, pctrl = _perc2_leadlag!(s.xllp, up, p.Tap, p.Tbp, dt)
    s.xllq, qctrl = _perc2_leadlag!(s.xllq, uq, p.Taq, p.Tbq, dt)
    # current command, limits, FracOn
    ip, iq = _perc2_command(s, p, s.vtf, pctrl, qctrl)
    ip *= frac
    iq = (iq + s.iqx_m) * frac
    # Tt lag (first call: no lag, the lag state starts at the command)
    tt_min = p.tt_floor_steps * dt
    s.tt_floor_applied |= p.Tt < tt_min
    if s.first_step
        s.ip_m, s.iq_m = ip, iq
        s.first_step = false
    else
        a = _perc2_alpha(max(p.Tt, tt_min), dt)
        s.ip_m += (ip - s.ip_m) * a
        s.iq_m += (iq - s.iq_m) * a
    end
    return perc2_output(s)
end

"""
    perc2_snapshot(s, vm) -> NamedTuple

Recorded quantities. `P` and `Q` are system-base pu at voltage `vm`;
`Ppu0 = P / P0` is the `P` column of the reference data. `mode` is 0 monitor,
1 ceased, 2 ramping; `level` is -1 (none), 0 momentary, 1 long, 2 frequency;
`nevents` is `length(perc2_events(s))`.
"""
function perc2_snapshot(s::PERC2State, vm::Real)
    Ip, Iq = perc2_output(s)
    P = vm * Ip
    Q = vm * Iq
    return (Vtf = s.vtf, ftf = s.ftf, FracOn = s.frac, P = P, Q = Q, Ppu0 = P / s.P0,
            ip_m = s.ip_m, iq_m = s.iq_m, mode = s.mode, level = s.level,
            tt_floor_applied = s.tt_floor_applied, nevents = length(s.events),
            Iqextra = s.iqx_m * s.mbase)
end

"""
    perc2_events(s) -> Vector{Tuple{Symbol,Float64,Symbol}}

Event history `(kind, t, check)`, kind ∈ `:trip` (load ceases), `:cease`
(every cease decision, including escalation while ceased),
`:reconnect_start`, `:reconnect_end`.
"""
perc2_events(s::PERC2State) = s.events

"""
    perc2_playback(p, P0, Q0, ts, vs, fs; dt = ts[2] - ts[1]) -> (snapshots, state)

Drive one PERC2 block with a prescribed bus voltage `vs` (pu) and frequency
`fs` (Hz) sampled at times `ts` (no network). The block is initialized at
`V0 = vs[1]` and `perc2_step!` is called at every `ts[k]`; `snapshots[k]` is
taken after that call with `vm = vs[k]` (row k of the reference data in
`test/perc2_reference`).
"""
function perc2_playback(p::PERC2Params, P0::Real, Q0::Real, ts::AbstractVector, vs::AbstractVector,
                        fs::AbstractVector; dt::Real = ts[2] - ts[1])
    length(ts) == length(vs) == length(fs) || throw(ArgumentError("ts, vs, fs must have equal length"))
    s = perc2_init(p, P0, Q0, vs[1])
    snaps = map(eachindex(ts)) do k
        perc2_step!(s, p, vs[k], fs[k], ts[k], dt)
        perc2_snapshot(s, vs[k])
    end
    return snaps, s
end