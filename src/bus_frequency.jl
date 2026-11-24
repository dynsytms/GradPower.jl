# Local bus-frequency estimate from the network voltage phasor.
#
# GradPower's network runs in the synchronous reference frame (ω_nom = 2π fnom),
# so the local frequency at a bus is fnom plus the rate of change of its voltage
# angle:
#
#     f_k = fnom (1 + Δθ_k / (ω_nom Δt_k)) = fnom + Δθ_k / (2π Δt_k),
#     Δθ_k = wrap(θ_k - θ_{k-1}) ∈ (-π, π],   Δt_k = t_k - t_{k-1} > 0,
#
# a backward difference over one accepted step (it lags the true frequency by
# about Δt/2). An optional first-order filter with time constant `Tfm` (exact
# discretization x += (u - x)(1 - exp(-Δt/Tfm))) smooths it; `Tfm = 0` returns
# the raw difference. Devices that apply their own filter (PERC2's Tf) should
# use `Tfm = 0`.
#
# The estimator lives outside z and is advanced only on ACCEPTED time steps,
# never inside Newton or residual evaluations. Events are applied between
# steps by a dt = 0 algebraic re-solve that makes the angle jump; that jump is
# not a frequency, so the hook must call `bus_frequency_reset!` after every
# event re-solve: it re-references θ to the post-event angle without changing
# the output. Calling `bus_frequency_update!` with Δt <= 0 does the same.
#
# Below `vmin` (a bolted fault) the angle is undefined: the last frequency is
# held and the reference is re-taken at the first sample with |V| >= vmin.

"""
    BusFrequencyEstimator(buses; fnom=60.0, Tfm=0.0, vmin=1e-3)

Local frequency estimate (Hz) at the internal bus indices `buses`, for one
scenario. Each bus keeps its previous angle and time, and outputs the angle
rate of change over the last accepted step, optionally filtered with time
constant `Tfm` (s). Use one estimator per scenario.

Call [`bus_frequency_init!`](@ref) at the initial state,
[`bus_frequency_update!`](@ref) after every accepted step (Δt > 0), and
[`bus_frequency_reset!`](@ref) after every dt = 0 event re-solve. The output
is `est.f[i]` for `buses[i]`.
"""
mutable struct BusFrequencyEstimator
    buses::Vector{Int}       # internal bus indices
    fnom::Float64            # nominal frequency (Hz)
    Tfm::Float64             # measurement filter time constant (s), 0 = raw
    vmin::Float64            # below this |V| (pu) the angle is not used
    theta::Vector{Float64}   # reference angle θ_{k-1} (rad)
    t::Vector{Float64}       # time of the reference angle (s)
    valid::BitVector         # reference angle is usable
    fraw::Vector{Float64}    # last unfiltered estimate (Hz)
    f::Vector{Float64}       # output (Hz)
end

function BusFrequencyEstimator(buses::AbstractVector{<:Integer}; fnom::Real=60.0,
                               Tfm::Real=0.0, vmin::Real=1e-3)
    fnom > 0 || throw(ArgumentError("BusFrequencyEstimator: fnom must be positive, got $fnom"))
    isfinite(Tfm) && Tfm >= 0 ||
        throw(ArgumentError("BusFrequencyEstimator: Tfm must be finite and >= 0, got $Tfm"))
    vmin >= 0 || throw(ArgumentError("BusFrequencyEstimator: vmin must be >= 0, got $vmin"))
    n = length(buses)
    return BusFrequencyEstimator(collect(Int, buses), Float64(fnom), Float64(Tfm), Float64(vmin),
                                 zeros(n), fill(NaN, n), falses(n), fill(Float64(fnom), n),
                                 fill(Float64(fnom), n))
end

# Wrap an angle difference into (-π, π].
@inline function _wrap_angle(d::Float64)
    d = rem2pi(d, RoundNearest)        # [-π, π]
    return d == -π ? Float64(π) : d
end

@inline _bus_v(z::AbstractVector, net_ptr::Int, bus::Int) =
    (z[net_ptr + 2 * (bus - 1) + 1], z[net_ptr + 2 * (bus - 1) + 2])

# Take (vr, vi) as the new reference angle at time t, if |V| allows it.
@inline function _bus_frequency_reference!(est::BusFrequencyEstimator, i::Int, vr, vi, t)
    if hypot(vr, vi) >= est.vmin
        est.theta[i] = atan(vi, vr)
        est.valid[i] = true
    else
        est.valid[i] = false
    end
    est.t[i] = t
    return nothing
end

"""
    bus_frequency_init!(est, z, net_ptr, t) -> est.f

Start the estimator at the initial state `z` (time `t`): the output is `fnom`
and the reference angles are those of `z`. `net_ptr = diff_dim + alg_dim` is
the offset of the network voltages in `z`.
"""
function bus_frequency_init!(est::BusFrequencyEstimator, z::AbstractVector, net_ptr::Int, t::Real)
    for (i, b) in enumerate(est.buses)
        vr, vi = _bus_v(z, net_ptr, b)
        bus_frequency_init!(est, i, vr, vi, t)
    end
    return est.f
end

"""
    bus_frequency_init!(est, i, vr, vi, t) -> fnom

Single-bus form of `bus_frequency_init!` for entry `i` of `est.buses`.
"""
function bus_frequency_init!(est::BusFrequencyEstimator, i::Int, vr::Real, vi::Real, t::Real)
    _bus_frequency_reference!(est, i, Float64(vr), Float64(vi), Float64(t))
    est.fraw[i] = est.fnom
    est.f[i] = est.fnom
    return est.f[i]
end

"""
    bus_frequency_update!(est, z, net_ptr, t) -> est.f

Advance every bus with the converged state `z` of an accepted step at time
`t`. Call once per accepted step, before that step's events. See the
single-bus form for the rules.
"""
function bus_frequency_update!(est::BusFrequencyEstimator, z::AbstractVector, net_ptr::Int, t::Real)
    for (i, b) in enumerate(est.buses)
        vr, vi = _bus_v(z, net_ptr, b)
        bus_frequency_update!(est, i, vr, vi, t)
    end
    return est.f
end

"""
    bus_frequency_update!(est, i, vr, vi, t) -> f_hz

Advance entry `i` of `est.buses` with its voltage `vr + j vi` (pu) at time
`t` (s) and return the estimate (Hz):

  * `Δt = t - t_prev > 0`, valid reference and `|V| >= vmin`:
    `fraw = fnom (1 + wrap(θ - θ_prev) / (2π fnom Δt))`, filtered by `Tfm`;
  * `Δt <= 0` (an algebraic re-solve at the same time): re-reference only,
    like [`bus_frequency_reset!`](@ref); the output is unchanged;
  * no valid reference (first sample after `|V| < vmin`): re-reference, the
    output is held;
  * `|V| < vmin`: the output is held and the reference invalidated.
"""
function bus_frequency_update!(est::BusFrequencyEstimator, i::Int, vr::Real, vi::Real, t::Real)
    vr, vi, t = Float64(vr), Float64(vi), Float64(t)
    dt = t - est.t[i]
    if !(dt > 0) || !est.valid[i] || hypot(vr, vi) < est.vmin
        _bus_frequency_reference!(est, i, vr, vi, t)
        return est.f[i]
    end
    th = atan(vi, vr)
    fnom = est.fnom
    fraw = fnom + _wrap_angle(th - est.theta[i]) / (2π * dt)
    est.fraw[i] = fraw
    est.theta[i] = th
    est.t[i] = t
    if est.Tfm > 0
        est.f[i] += (fraw - est.f[i]) * (-expm1(-dt / est.Tfm))
    else
        est.f[i] = fraw
    end
    return est.f[i]
end

"""
    bus_frequency_reset!(est, z, net_ptr) -> est

Re-reference every bus angle to the state `z` (the result of a dt = 0 event
re-solve) without changing the output or the reference time. The next
accepted step then differentiates from the post-event angle, so the angle
jump of the event is not read as frequency.
"""
function bus_frequency_reset!(est::BusFrequencyEstimator, z::AbstractVector, net_ptr::Int)
    for (i, b) in enumerate(est.buses)
        vr, vi = _bus_v(z, net_ptr, b)
        bus_frequency_reset!(est, i, vr, vi)
    end
    return est
end

"""
    bus_frequency_reset!(est, i, vr, vi) -> est

Single-bus form of `bus_frequency_reset!` for entry `i` of `est.buses`.
"""
function bus_frequency_reset!(est::BusFrequencyEstimator, i::Int, vr::Real, vi::Real)
    _bus_frequency_reference!(est, i, Float64(vr), Float64(vi), est.t[i])
    return est
end
