# PERC2Load residual, Jacobian and accepted-step hook.
#
# Inside Newton the load draws the system-base current
#
#     I = (Ip - j Iq) V/|V|,     V = vr + j vi,
#
# with Ip = p[pp], Iq = p[pp+1] held constant; the consumed power is
# V conj(I) = |V| (Ip + j Iq). As for ZIPLoad, the network rows hold the
# injected current, so I is SUBTRACTED from f[vr], f[vi] of the bus. Below
# |V| = PERC2_VMIN_DIRECTION the direction V/|V| is undefined: the direction
# of the last accepted step (p[pp+5], p[pp+6]) is used and the Jacobian
# contribution is zero.
#
# After every accepted step (and once at the initial state) the integrator
# calls `perc2_accept_step!`, which advances the bus-frequency estimate and
# the PERC2 logic with the converged state and writes the new held currents
# into p. They are never changed inside Newton, on failed solves or by the
# dt = 0 event re-solves.

const J_P2_VR_VR = 1
const J_P2_VR_VI = 2
const J_P2_VI_VR = 3
const J_P2_VI_VI = 4

const PERC2_VMIN_DIRECTION = 1e-6

# A held-current change above this (system-base pu) forces a Jacobian
# refactorization at the next solve. Noise-level changes on a flat run
# (~1e-11) must not, or the modified-Newton iterates differ from the no-PERC2
# system by more than the Newton tolerance.
const PERC2_REFACTOR_TOL = 1e-6

@inline function perc2_residual_batch!(f::AbstractArray, z::AbstractArray, p::AbstractArray,
                                       table::PERC2LoadTable, net_ptr::Int)
    table.n == 0 && return nothing
    @inbounds for k in 1:table.n
        table.online[k] || continue
        pp = Int(table.par_ptr[k])
        vr_idx = net_ptr + 2 * (Int(table.bus[k]) - 1) + 1
        vr = z[vr_idx]
        vi = z[vr_idx + 1]
        vm = sqrt(vr * vr + vi * vi)
        if vm >= PERC2_VMIN_DIRECTION
            cr = vr / vm
            ci = vi / vm
        else
            cr = p[pp + 5]
            ci = p[pp + 6]
        end
        Ip = p[pp]
        Iq = p[pp + 1]
        f[vr_idx]     -= Ip * cr + Iq * ci
        f[vr_idx + 1] -= Ip * ci - Iq * cr
    end
    return nothing
end

function perc2_jac_positions!(table::PERC2LoadTable, J::SparseMatrixCSC, net_ptr::Int)
    table.n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:table.n
        vr_idx = net_ptr + 2 * (Int(table.bus[k]) - 1) + 1
        vi_idx = vr_idx + 1
        table.jac_pos[k, J_P2_VR_VR] = _find_pos(J, rows, vr_idx, vr_idx)
        table.jac_pos[k, J_P2_VR_VI] = _find_pos(J, rows, vr_idx, vi_idx)
        table.jac_pos[k, J_P2_VI_VR] = _find_pos(J, rows, vi_idx, vr_idx)
        table.jac_pos[k, J_P2_VI_VI] = _find_pos(J, rows, vi_idx, vi_idx)
    end
    return nothing
end

@inline function perc2_jacobian_batch!(J::SparseMatrixCSC, z::AbstractArray, p::AbstractArray,
                                       table::PERC2LoadTable, net_ptr::Int)
    table.n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:table.n
        table.online[k] || continue
        pp = Int(table.par_ptr[k])
        vr_idx = net_ptr + 2 * (Int(table.bus[k]) - 1) + 1
        vr = z[vr_idx]
        vi = z[vr_idx + 1]
        vm = sqrt(vr * vr + vi * vi)
        vm < PERC2_VMIN_DIRECTION && continue
        # d(V/|V|)/dvr = (vi², -vr vi)/|V|³,  d(V/|V|)/dvi = (-vr vi, vr²)/|V|³
        inv3 = 1.0 / (vm * vm * vm)
        a = vi * vi * inv3
        b = -vr * vi * inv3
        c = vr * vr * inv3
        Ip = p[pp]
        Iq = p[pp + 1]
        nz[table.jac_pos[k, J_P2_VR_VR]] -= Ip * a + Iq * b
        nz[table.jac_pos[k, J_P2_VR_VI]] -= Ip * b + Iq * c
        nz[table.jac_pos[k, J_P2_VI_VR]] -= Ip * b - Iq * a
        nz[table.jac_pos[k, J_P2_VI_VI]] -= Ip * c - Iq * b
    end
    return nothing
end

# ------------------------------------------------------------------------
# Logic store and accepted-step hook (shared by every CPU integrator)
# ------------------------------------------------------------------------

"""
    perc2_ensure_scenarios!(table, M)

Make the per-scenario logic store of `table` hold at least `M` scenarios.
"""
function perc2_ensure_scenarios!(table::PERC2LoadTable, M::Int)
    while length(table.states) < M
        push!(table.states, PERC2State[])
        push!(table.freq, BusFrequencyEstimator(Int.(table.bus); fnom=_perc2_fnom(table)))
        push!(table.last_vm, zeros(table.n))
    end
    return table
end

function _perc2_fnom(table::PERC2LoadTable)
    table.n == 0 && return 60.0
    f = table.params[1].fnom
    all(q -> q.fnom == f, table.params) ||
        throw(ArgumentError("PERC2 loads with different fnom in one system are not supported"))
    return f
end

"""
    perc2_reset_scenario!(table, m, p)

Re-initialize the PERC2 logic of scenario `m` at its operating point
(`P0`, `Q0`, `V0` read from the parameter row `p`) and write the initial held
currents into `p`. Does not call `perc2_step!`.
"""
function perc2_reset_scenario!(table::PERC2LoadTable, m::Int, p::AbstractVector)
    perc2_ensure_scenarios!(table, m)
    st = table.states[m]
    resize!(st, table.n)
    for k in 1:table.n
        pp = Int(table.par_ptr[k])
        st[k] = perc2_init(table.params[k], p[pp + 2], p[pp + 3], p[pp + 4])
        p[pp], p[pp + 1] = perc2_output(st[k])
        table.last_vm[m][k] = p[pp + 4]
    end
    return nothing
end

"""
    perc2_init_scenario!(table, m, z, p, t, dt, net_ptr; log=nothing) -> changed::Bool

Start scenario `m` at the initial state `z` (time `t`): reset the PERC2 logic
and the bus-frequency estimator, then take the first `perc2_step!` at `t`
(which leaves a steady state unchanged and anchors the `Tt` lag). Writes the
held currents into `p`.
"""
function perc2_init_scenario!(table::PERC2LoadTable, m::Int, z::AbstractVector, p::AbstractVector,
                              t::Float64, dt::Float64, net_ptr::Int; log=nothing)
    table.n == 0 && return false
    perc2_ensure_scenarios!(table, m)
    table.freq[m] = BusFrequencyEstimator(Int.(table.bus); fnom=_perc2_fnom(table))
    perc2_reset_scenario!(table, m, p)
    bus_frequency_init!(table.freq[m], z, net_ptr, t)
    return _perc2_step_all!(table, m, z, p, t, dt, net_ptr, table.freq[m].f, log)
end

"""
    perc2_accept_step!(table, m, z, p, t, dt, net_ptr; log=nothing) -> changed::Bool

Accepted-step hook for scenario `m`: update the bus-frequency estimate with
the converged state `z` at time `t`, advance every online PERC2 block by one
step `dt` and write the new held currents (and the voltage direction) into
the parameter row `p`. Returns `true` if a held current changed by more than
`PERC2_REFACTOR_TOL` (the caller then refactors the Jacobian; smaller changes
are left to the modified-Newton convergence check). Call once per accepted
step, before that step's events. With `log::Vector`, pushes one
`(t, device, vm, snapshot...)` NamedTuple per PERC2 device.
"""
function perc2_accept_step!(table::PERC2LoadTable, m::Int, z::AbstractVector, p::AbstractVector,
                            t::Float64, dt::Float64, net_ptr::Int; log=nothing)
    table.n == 0 && return false
    f = bus_frequency_update!(table.freq[m], z, net_ptr, t)
    return _perc2_step_all!(table, m, z, p, t, dt, net_ptr, f, log)
end

"""
    perc2_event_resolved!(table, m, z, net_ptr)

Call after every converged dt = 0 event re-solve: re-references the bus
angles so the event's angle jump is not read as frequency. The held currents
and the PERC2 logic are not changed.
"""
function perc2_event_resolved!(table::PERC2LoadTable, m::Int, z::AbstractVector, net_ptr::Int)
    table.n == 0 && return nothing
    bus_frequency_reset!(table.freq[m], z, net_ptr)
    return nothing
end

function _perc2_step_all!(table::PERC2LoadTable, m::Int, z::AbstractVector, p::AbstractVector,
                          t::Float64, dt::Float64, net_ptr::Int, fbus::Vector{Float64}, log)
    changed = false
    st = table.states[m]
    @inbounds for k in 1:table.n
        pp = Int(table.par_ptr[k])
        vr_idx = net_ptr + 2 * (Int(table.bus[k]) - 1) + 1
        vr = z[vr_idx]
        vi = z[vr_idx + 1]
        vm = sqrt(vr * vr + vi * vi)
        table.last_vm[m][k] = vm
        if table.online[k]
            if vm >= PERC2_VMIN_DIRECTION
                p[pp + 5] = vr / vm
                p[pp + 6] = vi / vm
            end
            Ip, Iq = perc2_step!(st[k], table.params[k], vm, fbus[k], t, dt)
            changed |= abs(Ip - p[pp]) > PERC2_REFACTOR_TOL || abs(Iq - p[pp + 1]) > PERC2_REFACTOR_TOL
            p[pp] = Ip
            p[pp + 1] = Iq
        end
        log === nothing ||
            push!(log, merge((t = t, device = Int(table.device[k]), vm = vm, f_hz = fbus[k],
                              online = table.online[k]), perc2_snapshot(st[k], vm)))
    end
    return changed
end

"""
    perc2_states(sys; scenario=1) -> Vector{NamedTuple}

Current PERC2 quantities of every `PERC2Load` in `sys` (after
`initialize_dynamics!` or the last `integrate!`): device index, external bus,
id, online flag, events `perc2_events`, and `perc2_snapshot` at the last
accepted bus voltage (`Vtf`, `ftf`, `FracOn`, `P`, `Q`, `mode`, `level`, ...).
"""
function perc2_states(ps::PowerSystem; scenario::Integer=1)
    dyn = ps.dynamic
    dyn === nothing && return NamedTuple[]
    table = (dyn.layout::SimulationLayout).perc2load
    table.n == 0 && return NamedTuple[]
    length(table.states) >= scenario && length(table.states[scenario]) == table.n ||
        throw(ArgumentError("PERC2 states not initialized for scenario $scenario: run initialize_dynamics! first"))
    return NamedTuple[merge((device = Int(table.device[k]),
                             bus = ps.buses[table.bus[k]].i,
                             id = _normalize_id(dyn.devices[table.device[k]].dtype.id),
                             online = table.online[k],
                             events = copy(perc2_events(table.states[scenario][k]))),
                            perc2_snapshot(table.states[scenario][k], table.last_vm[scenario][k]))
                      for k in 1:table.n]
end

"""
    _check_no_perc2(ps, what)

Throw `ArgumentError` if `ps` has PERC2 loads: `what` does not support them.
"""
function _check_no_perc2(ps::PowerSystem, what::AbstractString)
    dyn = ps.dynamic
    (dyn === nothing || dyn.layout === nothing) && return nothing
    n = dyn.layout.perc2load.n
    n == 0 && return nothing
    throw(ArgumentError("$what does not support PERC2 loads ($n PERC2Load device(s) present); " *
                        "use the CPU integrators (integrate!)"))
end

# Varying parameter slots (Ip, Iq, dir_r, dir_i) saved/restored around a run.
const _PERC2_VARYING_OFFSETS = (0, 1, 5, 6)

_perc2_save_pvec(table::PERC2LoadTable, p::AbstractVector) =
    [p[Int(table.par_ptr[k]) + o] for k in 1:table.n for o in _PERC2_VARYING_OFFSETS]

function _perc2_restore_pvec!(table::PERC2LoadTable, p::AbstractVector, saved::Vector{Float64})
    j = 0
    for k in 1:table.n, o in _PERC2_VARYING_OFFSETS
        p[Int(table.par_ptr[k]) + o] = saved[j += 1]
    end
    return p
end
