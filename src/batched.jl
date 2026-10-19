# Multi-scenario integration.
#
# BatchedLayout holds the per-scenario inputs with scenario as the FIRST
# index (z[scenario, state], p[scenario, par], u[scenario, ctrl]) — the
# layout the GPU extension copies to the device for coalesced access — plus
# the Jacobian sparsity pattern shared by all scenarios.
#
# The CPU integrator (integrate_batched!) copies each scenario into its own
# contiguous workspace and runs the single-scenario Newton solve on it,
# threaded over scenarios.

# -----------------------------------------------------------------------
# BatchedLayout struct
# -----------------------------------------------------------------------

struct BatchedLayout
    M::Int                    # number of scenarios
    sys_dim::Int              # total system dimension per scenario
    diff_dim::Int
    alg_dim::Int
    nbus::Int
    z::Matrix{Float64}        # [M, sys_dim] initial / final state per scenario
    p::Matrix{Float64}        # [M, par_dim] parameters per scenario
    u::Matrix{Float64}        # [M, ctrl_dim]
    # Shared Jacobian sparsity pattern (CSC)
    J_colptr::Vector{Int}
    J_rowval::Vector{Int}
    # Shared admittance matrix
    ybus::SparseMatrixCSC{Float64,Int}
    # Injection metadata and uvec_idx routing (used by the GPU layout)
    inj_meta::InjectionMeta
    uvec_idx::Vector{Int64}
end

"""
    BatchedLayout(dp, ps, M)

`M` scenarios, each starting from `dp.zvec` with parameters `dp.pvec`. Edit
`bl.z[m, :]` / `bl.p[m, :]` to make the scenarios differ, then call
[`integrate_batched!`](@ref) (CPU) or build a `GpuBatchedLayout` from it.
"""
function BatchedLayout(dp::DynamicProblem, ps::PowerSystem, M::Int)
    dyn  = ps.dynamic::PowerSystemDynamics
    nbus = length(ps.buses)
    sys_dim  = dyn.diff_dim + dyn.alg_dim + 2*nbus
    L = dyn.layout::SimulationLayout

    @assert length(dp.zvec) == sys_dim
    @assert length(dp.pvec) == dyn.par_dim
    @assert length(dp.uvec) == dyn.ctrl_dim

    z = repeat(permutedims(dp.zvec), M)
    p = repeat(permutedims(dp.pvec), M)
    u = repeat(permutedims(dp.uvec), M)

    J_template = preallocate_jacobian(ps)
    return BatchedLayout(M, sys_dim, dyn.diff_dim, dyn.alg_dim, nbus, z, p, u,
                         copy(J_template.colptr), copy(J_template.rowval),
                         ps.network.ybus_real, InjectionMeta(L), dyn.uvec_idx)
end

# -----------------------------------------------------------------------
# Batched helper (GPU extension fallback path)
# -----------------------------------------------------------------------

"""
    beuler_batched_2d!(f, z, zold, diff_dim, is_diff, dt, M)

Apply backward Euler scaling: f[m,i] = z[m,i] - zold[m,i] - dt*f[m,i]
for differential state positions.
"""
@inline function beuler_batched_2d!(f::Matrix{Float64}, z::Matrix{Float64},
                                     zold::Matrix{Float64},
                                     is_diff::Union{Nothing,BitVector},
                                     diff_indices::Union{Nothing,Vector{Int}},
                                     diff_dim::Int, dt::Float64, M::Int)
    if diff_indices !== nothing
        @inbounds for i in diff_indices
            for m in 1:M
                f[m, i] = z[m, i] - zold[m, i] - dt * f[m, i]
            end
        end
    else
        @inbounds for i in 1:diff_dim
            for m in 1:M
                f[m, i] = z[m, i] - zold[m, i] - dt * f[m, i]
            end
        end
    end
    return nothing
end

# -----------------------------------------------------------------------
# integrate_batched!
#
# Each scenario has its own contiguous workspace and runs the same Newton
# solve as `integrate!` (factorization reuse + predictor). Scenarios are
# independent, so the per-step work is spread over threads; events are
# applied between steps, on the main task, to the shared system.
# -----------------------------------------------------------------------

mutable struct _ScenarioWorkspace
    z::Vector{Float64}
    zold::Vector{Float64}
    zprev::Vector{Float64}
    zpred::Vector{Float64}
    zanchor::Vector{Float64}   # Crank–Nicolson anchor z_k + h/2 f(z_k)
    fk::Vector{Float64}        # f(z_k) scratch for the anchor
    u::Vector{Float64}
    p::Vector{Float64}
    f::Vector{Float64}
    dx::Vector{Float64}
    zwork::Vector{Float64}
    J::SparseMatrixCSC{Float64,Int}
    fact::Base.RefValue{Any}
    reuse::JacobianReuse
    have_prev::Bool
end

function _ScenarioWorkspace(bl::BatchedLayout, m::Int, J_template::SparseMatrixCSC,
                            dyn::PowerSystemDynamics, ybus::SparseMatrixCSC,
                            L::SimulationLayout, dt::Float64,
                            linear_solver::AbstractDirectSolverBackend,
                            jacobian_reuse::Bool)
    n = bl.sys_dim
    z = Vector{Float64}(bl.z[m, :])
    u = Vector{Float64}(bl.u[m, :])
    p = Vector{Float64}(bl.p[m, :])
    J = copy(J_template)
    beuler_jac_batched!(J, z, u, p, dyn, ybus, L, dyn.diff_dim, dt)
    fact = Ref{Any}(ds_solver(linear_solver, J; tuned=true))
    return _ScenarioWorkspace(z, copy(z), zeros(n), zeros(n), zeros(n), zeros(n),
                              u, p, zeros(n), zeros(n),
                              zeros(n), J, fact, JacobianReuse(jacobian_reuse), false)
end

# One implicit solve for scenario workspace `w`: a time step (dt > 0, with the
# predictor) or the algebraic re-solve after an event (dt == 0). `integrator`
# selects the step method (see `_cn_anchor!`).
function _solve_scenario!(w::_ScenarioWorkspace, ps::PowerSystem, dt::Float64,
                          tol::Float64, predictor::Bool,
                          integrator::Symbol=:backward_euler)
    copyto!(w.zold, w.z)
    anchor = w.zold
    h = dt
    if dt > 0.0 && integrator === :crank_nicolson
        dyn = ps.dynamic::PowerSystemDynamics
        _cn_anchor!(w.zanchor, w.fk, w.zold, w.u, w.p, dyn, (ps.network::Network).ybus_real,
                    dyn.layout::SimulationLayout, dt)
        anchor = w.zanchor
        h = 0.5 * dt
    end
    guess = w.z
    if dt > 0.0 && predictor && w.have_prev
        guess = w.zpred
        @inbounds for i in eachindex(guess)
            guess[i] = 2.0*w.zold[i] - w.zprev[i]
        end
    end
    dt > 0.0 && copyto!(w.zprev, w.zold)
    ok = newton_step!(guess, w.f, w.J, w.fact, anchor, w.u, w.p, ps, h;
                      tol=tol, dx=w.dx, zwork=w.zwork, reuse=w.reuse)
    guess === w.z || copyto!(w.z, guess)
    w.have_prev = dt > 0.0
    return ok
end

# Run `f(m)` for m = 1:M, threaded when more than one thread is available.
function _foreach_scenario(f, M::Int, threaded::Bool)
    if threaded && Threads.nthreads() > 1 && M > 1
        Threads.@threads :dynamic for m in 1:M
            f(m)
        end
    else
        for m in 1:M
            f(m)
        end
    end
    return nothing
end

"""
    integrate_batched!(bl, ps, tf; dt=1/120, newton_tol=1e-10, linear_solver=KLUBackend(),
                       jacobian_reuse=true, predictor=true, threaded=true) -> (tvec, trajs)

Integrate the `bl.M` scenarios of `bl` (each with its own initial state
`bl.z[m, :]` and parameters `bl.p[m, :]`) through the events registered on
`ps`. Every scenario runs the same Newton solve as [`integrate!`](@ref) with
`solver=:monolithic`, so a scenario's trajectory equals `integrate!` with the
same `z`, `p` and options. Scenarios run on `Threads.nthreads()` threads when
`threaded=true`. `integrator` is `:backward_euler` (default) or
`:crank_nicolson`, as in `integrate!`. Returns `trajs[m]`, the `(sys_dim, nsteps+1)` trajectory of
scenario `m`; the final states are written back to `bl.z`.

`save_every`, `save_idxs` and `save_eltype` select the stored output as in
[`integrate!`](@ref) (same for every scenario). `step_callback(m, k, t, z)` is
called for scenario `m` with its state after the initial state (`k = 1`) and
after every accepted step; with `threaded=true` it runs on worker threads,
concurrently for different `m`, so it must only touch per-scenario data.
"""
function integrate_batched!(bl::BatchedLayout, ps::PowerSystem, tf::Float64;
                             dt::Float64=1.0/120.0, newton_tol::Float64=1e-10,
                             linear_solver::AbstractDirectSolverBackend=KLUBackend(),
                             jacobian_reuse::Bool=true, predictor::Bool=true,
                             threaded::Bool=true, save_every::Integer=1,
                             save_idxs=Colon(),
                             save_eltype::Type{<:AbstractFloat}=Float64,
                             step_callback=nothing,
                             integrator::Symbol=:backward_euler)
    _check_integrator(integrator)
    dyn = ps.dynamic::PowerSystemDynamics
    L = dyn.layout::SimulationLayout
    ybus = (ps.network::Network).ybus_real
    M = bl.M

    nsteps, tvec = _time_grid(tf, dt)

    events = dyn.events
    event_schedule = Tuple{Int,Int,Symbol}[]
    for (ei, ev) in enumerate(events)
        push!(event_schedule, (Int(round(ev.ton / dt)), ei, :on))
        push!(event_schedule, (Int(round(ev.toff / dt)), ei, :off))
    end
    load_steps = dyn.load_step_events
    for (li, lev) in enumerate(load_steps)
        _check_event_time(lev.t, dt, "LoadStepEvent")
        push!(event_schedule, (Int(round(lev.t / dt)), li, :load_step))
    end
    # Removals (device disconnects, line trips) are the same for every
    # scenario: they change the shared system between steps.
    disconnect_events = dyn.disconnect_events
    for (di, dev) in enumerate(disconnect_events)
        _check_event_time(dev.ton, dt, "DisconnectDeviceEvent")
        push!(event_schedule, (Int(round(dev.ton / dt)), di, :disconnect))
    end
    trip_events = dyn.trip_events
    for (ti, tev) in enumerate(trip_events)
        _check_event_time(tev.ton, dt, "TripLineEvent")
        push!(event_schedule, (Int(round(tev.ton / dt)), ti, :trip))
        isfinite(tev.toff) && push!(event_schedule, (Int(round(tev.toff / dt)), ti, :reclose))
    end
    sort!(event_schedule, by = x -> x[1])

    # preallocate_jacobian writes the layout's position caches, so it runs
    # once here (not per thread); each scenario gets a copy of the pattern.
    J_template = preallocate_jacobian(ps)
    ws = Vector{_ScenarioWorkspace}(undef, M)
    _foreach_scenario(M, threaded) do m
        ws[m] = _ScenarioWorkspace(bl, m, J_template, dyn, ybus, L,
                                   _step_coefficient(integrator, dt),
                                   linear_solver, jacobian_reuse)
    end

    # One recorder (own output buffer) per scenario: threads never share one.
    recs = [_Recorder(bl.sys_dim, nsteps, save_idxs, save_every, save_eltype) for _ in 1:M]
    for m in 1:M
        _record!(recs[m], 1, ws[m].z)
        step_callback === nothing || step_callback(m, 1, tvec[1], ws[m].z)
    end
    ok = trues(M)
    # Load steps scale each scenario's own initial parameters (bl.p is not
    # modified: the workspaces hold copies).
    p0s = isempty(load_steps) ? nothing : [copy(w.p) for w in ws]
    saved = _snapshot_event_state(ps)

    try
    sched_idx = 1
    for k in 1:nsteps
        _foreach_scenario(M, threaded) do m
            ok[m] = _solve_scenario!(ws[m], ps, dt, newton_tol, predictor, integrator)
            _record!(recs[m], k + 1, ws[m].z)
            step_callback === nothing || step_callback(m, k + 1, tvec[k + 1], ws[m].z)
        end
        all(ok) || check_newton(false, tvec[k + 1],
            "batched time step (scenarios $(findall(!, ok)))")

        any_event = false
        while sched_idx <= length(event_schedule) && event_schedule[sched_idx][1] == k
            _, idx, action = event_schedule[sched_idx]
            if action === :on
                activate!(events[idx])
            elseif action === :off
                deactivate!(events[idx])
            elseif action === :load_step
                for m in 1:M
                    _apply_load_step!(ws[m].p, p0s[m], dyn, load_steps[idx])
                end
            elseif action === :disconnect
                _set_device_offline!(dyn, disconnect_events[idx].device_idx)
            elseif action === :trip
                _apply_trip_line!(ybus, trip_events[idx])
            elseif action === :reclose
                _apply_reclose_line!(ybus, trip_events[idx])
            end
            any_event = true
            sched_idx += 1
        end

        if any_event
            _foreach_scenario(M, threaded) do m
                ws[m].reuse.dt = NaN   # the equations changed
                ok[m] = _solve_scenario!(ws[m], ps, 0.0, newton_tol, predictor)
            end
            all(ok) || check_newton(false, tvec[k + 1],
                "batched dt=0 event re-solve (scenarios $(findall(!, ok)))")
        end
    end
    finally
        _restore_event_state!(ps, saved)
    end

    for m in 1:M
        bl.z[m, :] .= ws[m].z
    end
    return _saved_times(tvec, recs[1].every), [r.traj for r in recs]
end
