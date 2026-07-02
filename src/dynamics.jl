include("generators.jl")
include("loads.jl")
include("governors.jl")
include("exciters.jl")
include("stabilizers.jl")
include("static_gen.jl")

function initialize_device(
        device::DynamicDevice,
        i::Int64,
        map::DynamicMap,
        ps::PowerSystem,
        z::AbstractArray,
        diff_dim::Int64,
        u::AbstractArray,
        p::AbstractArray
    )

    # retrieve pointers and sizes
    diff_ptr = device.diff_ptr
    alg_ptr = device.alg_ptr
    ctrl_ptr = device.ctrl_ptr
    par_ptr = device.par_ptr

    diff_size = device.dtype.diff_size
    alg_size = device.dtype.alg_size
    ctrl_size = device.dtype.ctrl_size
    par_size = device.dtype.par_size

    # parameter view
    pview = @view(p[par_ptr:par_ptr+par_size-1])

    # find generator in static system and retrieve power injection.
    gen = ps.gens[map.gen[i]]
    pg = gen.psch
    qg = gen.qsch

    # retrieve voltage magnitude and angle.
    vm = ps.buses[map.bus[i]].v0m
    va = ps.buses[map.bus[i]].v0a

    # allocate initializing vector.
    # NOTE: I would prefer not to allocate. Not sure how to do it more efficiently since
    # I need to separate zvec and uvec.
    xinit = zeros(Float64, device.dtype.diff_size + device.dtype.alg_size + device.dtype.ctrl_size)
    # initial guess and initialization
    initial_guess!(xinit, pview, pg, qg, vm, va, device.dtype)
    rhs_fun!(f, x) = initialize_dynamics!(f, x, pview, pg, qg, vm, va, device.dtype)
    sol = nlsolve(rhs_fun!, xinit, ftol=1e-12, iterations=30, autodiff = :forward)

    # warning if not converged
    if !sol.f_converged
        println("Warning: initialization of device $(i) did not converge.")
    end

    # Verify residual at solution. xinit is reused as a scratch buffer here.
    rhs_fun!(xinit, sol.zero)
    @assert maximum(abs, xinit) < 1e-9 "Init residual $(maximum(abs, xinit)) exceeds 1e-9 for device $i of type $(typeof(device.dtype))"
    xinit .= sol.zero
    # copy to zvec and uvec. diff_ptr is the absolute z-index; the alg
    # states live at diff_dim + alg_ptr (absolute z-index after cluster-
    # contiguous reordering).
    z[diff_ptr:diff_ptr+diff_size-1] .= xinit[1:diff_size]
    alg_global = diff_dim + alg_ptr
    z[alg_global:alg_global+alg_size-1] .= xinit[diff_size+1:diff_size+alg_size]
    u[ctrl_ptr:ctrl_ptr+ctrl_size-1] .= xinit[diff_size+alg_size+1:diff_size+alg_size+ctrl_size]

    # Per-device post-init hook: governors / exciters may have
    # initialization-derived parameters (pref, vref) whose converged values
    # come out of nlsolve as extra unknowns past the diff/alg/ctrl slots.
    # `extract_init_params!` is the no-op default; concrete devices override.
    extract_init_params!(device.dtype, sol.zero, p, par_ptr)
end

# Default no-op. Concrete devices (IEESGO, etc.) override to capture
# extra unknowns from `sol.zero` into per-device fields AND mirror into
# the shared pvec slot so the kernel SoA snapshot is correct.
extract_init_params!(::AbstractDeviceType, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int) = nothing

# Default no-op set_ratio!. Devices whose parameters live on the
# machine base (Genrou, TGOV1) override.
set_ratio!(::AbstractDeviceType, ratio::Float64) = nothing

# IEESGO: pref is the 7th unknown (after 5 diff + 1 alg + 1 ctrl = 6;
# initial_guess fills it at slot 7). Store on the device struct AND in
# pvec slot 12 so kernels reading from SoA see the correct value.
function extract_init_params!(dtype::IEESGO, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    dtype.pref = sol_zero[7]
    p[par_ptr + 11] = dtype.pref   # slot 12, zero-based offset 11
    return nothing
end

# Genrou: stash post-PF e_fd (the 11th unknown — diff 6, alg 4, then
# ctrl[1]=e_fd, ctrl[2]=p_m) so attached exciters can read it during
# their own init. Not mirrored into pvec; e_fd is not a Genrou parameter.
function extract_init_params!(dtype::Genrou, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    dtype.e_fd0 = sol_zero[11]
    return nothing
end

# TGOV1: pref is the 4th unknown (after 2 diff + 1 alg + 1 ctrl). Slot 8 in pvec.
function extract_init_params!(dtype::TGOV1, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    dtype.pref = sol_zero[4]
    p[par_ptr + 7] = dtype.pref    # slot 8, zero-based offset 7
    return nothing
end

# SEXS: vref is the 3rd unknown (2 diff + 1 init slot). Slot 7 in pvec.
function extract_init_params!(dtype::SEXS, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    dtype.vref = sol_zero[3]
    p[par_ptr + 6] = dtype.vref    # slot 7, zero-based offset 6
    return nothing
end

# ESDC1A: vref is computed in closed form by `initial_guess!` and stashed
# on `dtype.vref` (and mirrored into pvec slot 10). Re-mirror here so the
# refresh path can pick it up unconditionally.
function extract_init_params!(dtype::ESDC1A, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    p[par_ptr + 9] = dtype.vref    # slot 10, zero-based offset 9
    return nothing
end

function initialize_dynamics!(dp::DynamicProblem, ps::PowerSystem)

    z = dp.zvec
    p = dp.pvec
    u = dp.uvec

    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    ctrl_dim = ps.dynamic.ctrl_dim
    par_dim = ps.dynamic.par_dim
    nbus = length(ps.buses)

    # initialize voltages
    for i in 1:nbus
        vm = ps.buses[i].v0m
        va = ps.buses[i].v0a
        z[diff_dim + alg_dim + 2*(i - 1) + 1] = vm*cos(va)
        z[diff_dim + alg_dim + 2*(i - 1) + 2] = vm*sin(va)
    end

    # TODO: write initialization function for load
    # set initial voltage magnitude in ZIPLoad devices.
    for (i, device) in enumerate(ps.dynamic.devices)
        if device.dtype isa ZIPLoad
            device.dtype.v0mag = ps.buses[ps.dynamic.map.bus[i]].v0m
            yload = (device.dtype.pinj + 1im*device.dtype.qinj)/(device.dtype.v0mag^2.0)
            device.dtype.yreal = real(yload)
            device.dtype.yimag = imag(yload)
        end
    end
    
    # fill parameter vector.
    # TODO: I need to re-think this design. To fill pvec I first need to set the load parameters with the
    # updated power-flow solution. But Ideally I should initialize all the devices after I fill the pvec.
    # In this case the load becomes an special case.
    for (i, device) in enumerate(ps.dynamic.devices)
        fill_pvec!(@view(dp.pvec[device.par_ptr:device.par_ptr+device.dtype.par_size-1]), device.dtype)
    end

    map = ps.dynamic.map

    # Initialize generators
    for (i, device) in enumerate(ps.dynamic.devices)
        if device.dtype isa AbstractGeneratorType
            initialize_device(device, i, map, ps, z, diff_dim, u, p)
        end
    end

    # Initialize StaticGenerators: aggregate P, Q from their gen_idxs into
    # device.p0, device.q0; populate alg state(s) (q for PV; p,q for SLACK).
    for (i, device) in enumerate(ps.dynamic.devices)
        device.dtype isa StaticGenerator || continue
        sg = device.dtype
        psum = 0.0
        qsum = 0.0
        for gi in sg.gen_idxs
            psum += ps.gens[gi].psch
            qsum += ps.gens[gi].qsch
        end
        sg.p0 = psum
        sg.q0 = qsum
        # Write alg states using absolute z-index (diff_dim + alg_ptr).
        alg_global = diff_dim + device.alg_ptr
        if sg.bus_type == 2          # PV: alg state is q
            z[alg_global] = qsum
        elseif sg.bus_type == 3      # SLACK: alg states are p, q
            z[alg_global]     = psum
            z[alg_global + 1] = qsum
        end
        # mirror p0, q0 into pvec so the kernel reads correct values
        p[device.par_ptr]     = psum
        p[device.par_ptr + 1] = qsum
    end

    # Pre-controller hook: stash each Genrou's post-PF e_fd0 onto any
    # attached SEXS's `vref` slot (used as scratch in `initial_guess!`
    # / `initialize_dynamics!`; `extract_init_params!(::SEXS)`
    # overwrites it with the converged vref after the controller solve).
    for (i, device) in enumerate(ps.dynamic.devices)
        device.dtype isa SEXS || continue
        # Match to the Genrou by (bus, id).
        gen_id = _normalize_id(device.dtype.id)
        for cand in ps.dynamic.devices
            if cand.dtype isa Genrou &&
               cand.dtype.bus == device.dtype.bus &&
               _normalize_id(cand.dtype.id) == gen_id
                device.dtype.vref = cand.dtype.e_fd0
                break
            end
        end
    end

    # Same hook for ESDC1A — its `initial_guess!` reads e_fd0 from
    # `dtype.vref` as scratch and overwrites it with the computed vref.
    for (i, device) in enumerate(ps.dynamic.devices)
        device.dtype isa ESDC1A || continue
        gen_id = _normalize_id(device.dtype.id)
        for cand in ps.dynamic.devices
            if cand.dtype isa Genrou &&
               cand.dtype.bus == device.dtype.bus &&
               _normalize_id(cand.dtype.id) == gen_id
                device.dtype.vref = cand.dtype.e_fd0
                break
            end
        end
    end

    # Initialize controllers. Generator needs to be initialized before
    # the controllers to set the initial power injection.
    for (i, device) in enumerate(ps.dynamic.devices)
        if device.dtype isa AbstractGenControlType
            initialize_device(device, i, map, ps, z, diff_dim, u, p)
        end
    end

    # Re-snapshot ZIPLoad table columns now that the power-flow solution
    # has populated v0mag/yreal/yimag on the device structs. Batched
    # kernels read from the table, so the snapshot must reflect post-init
    # values, not the build-time zeros. IEESGO has the same need for
    # `pref` (init-derived parameter).
    if ps.dynamic.layout !== nothing
        refresh_zipload_table!(ps.dynamic)
        refresh_ieesgo_table!(ps.dynamic)
        refresh_tgov1_table!(ps.dynamic)
        refresh_sexs_table!(ps.dynamic)
        refresh_esdc1a_table!(ps.dynamic)
        refresh_ieeest_table!(ps.dynamic)
        refresh_static_gen_table!(ps.dynamic)
    end
end

function rhs_fun!(f::AbstractArray, z::AbstractArray, u::AbstractArray, p::AbstractArray, sys::PowerSystem)
    # Batched kernels are the sole hot path. The function barrier
    # specializes on the concrete `PowerSystemDynamics`/`Network`/
    # `SimulationLayout` types — the `Union{Nothing,...}` fields on
    # `PowerSystem` would otherwise force every `getproperty` to box,
    # costing ~2.2 KiB per call. The barrier resolves the Unions once
    # at the entry, then dispatches to the batched implementation.
    dyn  = sys.dynamic::PowerSystemDynamics
    net  = sys.network::Network
    L    = dyn.layout::SimulationLayout
    _rhs_fun_batched!(f, z, u, p, dyn, net.ybus_real, L)
end

@noinline function _rhs_fun_batched!(f::AbstractArray, z::AbstractArray, u::AbstractArray,
                                    p::AbstractArray, dyn::PowerSystemDynamics,
                                    ybus::SparseMatrixCSC, L::SimulationLayout,
                                    log::Union{Nothing,SolverLog}=nothing)
    diff_dim = dyn.diff_dim
    alg_dim  = dyn.alg_dim
    net_ptr  = diff_dim + alg_dim
    v  = @view z[net_ptr+1:end]
    fv = @view f[net_ptr+1:end]
    if log !== nothing
        _t0 = time_ns()
        mul!(fv, ybus, v, -1.0, 0.0)
        log.ybus_mul_ns += time_ns() - _t0
        log.ybus_mul_count += 1
    else
        mul!(fv, ybus, v, -1.0, 0.0)
    end

    # Apply cross-device control routing: u[i] ← z[uvec_idx[i]] for
    # each wired ctrl slot. uvec_idx[i] == 0 means "not wired" (slot
    # stays at its init value of 0). Without this, controllers
    # (governor p_m, exciter e_fd) read stale init values throughout
    # the integration — Genrou's p_m would be frozen at t=0.
    _apply_uvec_routing!(u, z, dyn.uvec_idx)

    genrou_residual_batch!(f, z, u, p, L.genrou, diff_dim, net_ptr)
    ieesgo_residual_batch!(f, z, p, L.ieesgo, diff_dim)
    tgov1_residual_batch!(f, z, p, L.tgov1, diff_dim)
    ieeest_residual_batch!(f, z, p, L.ieeest, diff_dim)
    sexs_residual_batch!(f, z, p, L.sexs)
    esdc1a_residual_batch!(f, z, p, L.esdc1a)
    zipload_residual_batch!(f, z, p, L.zipload, net_ptr)
    static_gen_residual_batch!(f, z, p, L.static_gen)

    _apply_events_fun!(f, v, dyn.events, net_ptr)
    return nothing
end

@inline function _apply_uvec_routing!(u::AbstractArray, z::AbstractArray,
                                       uvec_idx::Vector{Int64})
    @inbounds for i in eachindex(uvec_idx)
        src = uvec_idx[i]
        if src != 0
            u[i] = z[src]
        end
    end
    return nothing
end

@inline function _apply_events_fun!(f::AbstractArray, v::AbstractArray,
                                     events::Vector{ContingencyEvent}, net_ptr::Int)
    @inbounds for event in events
        if event.status
            bus = event.bus
            vr = v[2*(bus-1)+1]
            vi = v[2*(bus-1)+2]
            yfault = 1.0/event.rfault
            f[net_ptr+2*(bus-1)+1] -= yfault*vr
            f[net_ptr+2*(bus-1)+2] -= yfault*vi
        end
    end
    return nothing
end

# NOTE: TODO: Implement different objective types using multiple dispatch.
function functional(z::AbstractArray, u::AbstractArray, p::AbstractArray, sys::PowerSystem)
    idxs = gen_speeds(sys)
    val = 0.0
    for (i, idx) in enumerate(idxs)
        val += z[idx]^2.0
    end
    return val
end

function beuler!(
    f::AbstractVector,
    z::AbstractVector,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    sys::PowerSystem,
    diff_dim::Int64,
    dt::Float64
)
    rhs_fun!(f, z, u, p, sys)
    di = sys.dynamic.diff_indices
    if di !== nothing
        @inbounds for i in di
            f[i] = z[i] - zold[i] - dt*f[i]
        end
    else
        @inbounds for i = 1:diff_dim
            f[i] = z[i] - zold[i] - dt*f[i]
        end
    end
end

# Typed variant — caller pre-extracts dyn / ybus / layout once so the
# Union{Nothing,...} fields on `PowerSystem` are not re-loaded each
# Newton iteration. Used by `newton_step!` for the per-iteration hot
# path; zero-alloc on its own.
@inline function beuler_batched!(
    f::AbstractVector,
    z::AbstractVector,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    dyn::PowerSystemDynamics,
    ybus::SparseMatrixCSC,
    L::SimulationLayout,
    diff_dim::Int64,
    dt::Float64,
    log::Union{Nothing,SolverLog}=nothing,
)
    _rhs_fun_batched!(f, z, u, p, dyn, ybus, L, log)
    di = dyn.diff_indices
    if di !== nothing
        @inbounds for i in di
            f[i] = z[i] - zold[i] - dt*f[i]
        end
    else
        @inbounds for i = 1:diff_dim
            f[i] = z[i] - zold[i] - dt*f[i]
        end
    end
    return nothing
end

function beuler_jac!(
    J::SparseMatrixCSC,
    z::AbstractVector,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    sys::PowerSystem,
    diff_dim::Int64,
    dt::Float64
)
    # set all elements to zero
    fill!(J.nzval, 0.0)

    # evaluate Jacobian
    rhs_jac!(J, z, u, p, sys)

    # scale for backward Euler
    isd = sys.dynamic.is_diff
    if isd !== nothing
        _jacobian_beuler_indices!(J, isd, dt)
    else
        _jacobian_beuler!(J, diff_dim, dt)
    end
end

# Typed variant — same rationale as `beuler_batched!`.
@inline function beuler_jac_batched!(
    J::SparseMatrixCSC,
    z::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    dyn::PowerSystemDynamics,
    ybus::SparseMatrixCSC,
    L::SimulationLayout,
    diff_dim::Int64,
    dt::Float64,
)
    fill!(J.nzval, 0.0)
    _rhs_jac_batched!(J, z, u, p, dyn, ybus, L)
    isd = dyn.is_diff
    if isd !== nothing
        _jacobian_beuler_indices!(J, isd, dt)
    else
        _jacobian_beuler!(J, diff_dim, dt)
    end
    return nothing
end

function _jacobian_beuler!(J::SparseMatrixCSC, NDIFFEQ::Int, h::Float64)
    # Iterating through each column
    for col = 1:size(J, 2)
        # Flag to check if diagonal element for the column is found
        diagonal_found = false

        # Iterating through the non-zero elements in each column
        for row_index in nzrange(J, col)
            row = rowvals(J)[row_index]

            # Update values if the row index is less or equal to NDIFFEQ
            if row <= NDIFFEQ
                J.nzval[row_index] *= -h

                # Update diagonal element
                if row == col
                    J.nzval[row_index] += 1.0
                    diagonal_found = true
                end
           end
        end

        # If diagonal element was not found and col is within NDIFFEQ, then add it
        if !diagonal_found && col <= NDIFFEQ
            @warn "Diagonal element not found for column $col. Adding it."
            J[col, col] += 1.0
        end
    end
end

"""
    _jacobian_beuler_indices!(J, is_diff, h)

Backward-Euler Jacobian scaling for cluster-contiguous layout.
`is_diff` is a precomputed BitVector: `is_diff[i] == true` iff z[i] is
a differential state. For each such position i, the row is scaled:
J[i,:] *= -h and J[i,i] += 1.0.

Allocation-free: `is_diff` is precomputed once during `reorder_state!`.
"""
function _jacobian_beuler_indices!(J::SparseMatrixCSC, is_diff::BitVector, h::Float64)
    n_da = length(is_diff)
    @inbounds for col = 1:size(J, 2)
        diagonal_found = false
        for row_index in nzrange(J, col)
            row = rowvals(J)[row_index]
            if row <= n_da && is_diff[row]
                J.nzval[row_index] *= -h
                if row == col
                    J.nzval[row_index] += 1.0
                    diagonal_found = true
                end
            end
        end
        if !diagonal_found && col <= n_da && is_diff[col]
            @warn "Diagonal element not found for diff column $col. Adding it."
            J[col, col] += 1.0
        end
    end
end

function integrate!(
    dp::DynamicProblem,
    ps::PowerSystem,
    tf::Float64;
    dt::Float64=(1.0/120.0),
    verbose::Bool=false,
    log::Union{Nothing,SolverLog}=nothing,
    solver::Symbol=:monolithic,
    newton_tol::Float64=1e-10,
    newton_norm::Symbol=:inf,
)
    # TODO: here we should have some checks to ensure that the problem is initialized

    # calculate number of steps
    nsteps = Int(round(tf/dt))
    tvec = collect(0:dt:tf)

    # retrieve events — build sorted schedule of (step, event_index, action)
    # tuples so the time loop can process all events at each step in O(1).
    events = ps.dynamic.events
    disconnect_events = ps.dynamic.disconnect_events
    trip_events = ps.dynamic.trip_events
    # Schedule: each entry is (step, event_index, action)
    #   :on/:off    — ContingencyEvent activate/deactivate
    #   :disconnect — DisconnectDeviceEvent
    #   :trip       — TripLineEvent
    event_schedule = Tuple{Int,Int,Symbol}[]
    for (ei, ev) in enumerate(events)
        push!(event_schedule, (Int(round(ev.ton / dt)), ei, :on))
        push!(event_schedule, (Int(round(ev.toff / dt)), ei, :off))
    end
    for (di, dev) in enumerate(disconnect_events)
        push!(event_schedule, (Int(round(dev.ton / dt)), di, :disconnect))
    end
    for (ti, tev) in enumerate(trip_events)
        push!(event_schedule, (Int(round(tev.ton / dt)), ti, :trip))
        if isfinite(tev.toff)
            push!(event_schedule, (Int(round(tev.toff / dt)), ti, :reclose))
        end
    end
    sort!(event_schedule, by = x -> x[1])

    # retrieve sizes
    nbus = length(ps.buses)
    system_size = ps.dynamic.diff_dim + ps.dynamic.alg_dim + 2*nbus
    @assert system_size == length(dp.zvec)

    # allocate solution trajectory.
    traj = zeros(Float64, system_size, nsteps+1)

    # allocate temporary vectors
    zold = zeros(Float64, system_size)
    # Newton scratch — reused across every Newton call so the inner
    # loop allocates nothing per iteration.
    dx_buf   = zeros(Float64, system_size)
    zwork    = zeros(Float64, system_size)
    verbose && println("Integrating from t = 0 s to t = $tf s with dt = $dt s.")
    # newton parameters — default tolerance 1e-10. Looser tolerance
    # leaves residual algebraic state error that gets amplified through
    # fault transients (gen i_q/i_d off by ~1e-2 if tol is 1e-9, vs
    # ~1e-10 with tol=1e-10).
    ftol = newton_tol
    max_iter = 30

    # initial condition
    zold .= dp.zvec
    traj[:,1] .= dp.zvec

    # initialize residual and Jacobian
    f0 = zeros(Float64, system_size)
    J0 = preallocate_jacobian(ps)
    beuler_jac!(J0, zold, zold, dp.uvec, dp.pvec, ps, ps.dynamic.diff_dim, dt)

    # pre-factorization
    fact = klu(J0)
    fact.common.scale = 0
    fact.common.btf = 0
    fact.common.ordering = 1
    fact.common.tol = 1e-3

    # Schur workspace (preallocated once, reused every step)
    use_schur = solver === :schur
    use_schur_gmres = solver === :schur_gmres
    sw = use_schur ? SchurWorkspace(ps) : nothing
    gsw = use_schur_gmres ? GmresSchurWorkspace(ps, zold, dp.pvec, dt) : nothing

    # time loop
    sched_idx = 1  # pointer into sorted event_schedule
    for k in 1:nsteps
        verbose && println("Time-stepping. t = $(tvec[k]) s.")
        if use_schur
            newton_step_schur!(zold, f0, J0, sw, zold, dp.uvec, dp.pvec, ps, dt, verbose=verbose, tol=ftol, zwork=zwork, log=log, newton_norm=newton_norm)
        elseif use_schur_gmres
            newton_step_schur_gmres!(zold, f0, J0, gsw, zold, dp.uvec, dp.pvec, ps, dt, verbose=verbose, tol=ftol, zwork=zwork, log=log, newton_norm=newton_norm)
        else
            newton_step!(zold, f0, J0, fact, zold, dp.uvec, dp.pvec, ps, dt, verbose=verbose, jac_verify=false, tol=ftol, dx=dx_buf, zwork=zwork, log=log, newton_norm=newton_norm)
        end
        traj[:,k+1] .= zold

        # process all events scheduled at this step
        any_event = false
        any_topology_change = false
        while sched_idx <= length(event_schedule) && event_schedule[sched_idx][1] == k
            step_k, idx, action = event_schedule[sched_idx]
            if action === :on
                activate!(events[idx])
                verbose && println("Fault $idx activated at t = $(events[idx].ton) s. Bus $(events[idx].bus).")
            elseif action === :off
                deactivate!(events[idx])
                verbose && println("Fault $idx deactivated at t = $(events[idx].toff) s.")
            elseif action === :disconnect
                dev_ev = disconnect_events[idx]
                _set_device_offline!(ps.dynamic, dev_ev.device_idx)
                verbose && println("Disconnect $idx: device $(dev_ev.device_idx) disconnected at t = $(dev_ev.ton) s.")
            elseif action === :trip
                tev = trip_events[idx]
                _apply_trip_line!(ps.network.ybus_real, tev)
                any_topology_change = true
                verbose && println("Trip $idx: line $(tev.from_bus)-$(tev.to_bus) tripped at t = $(tev.ton) s.")
            elseif action === :reclose
                tev = trip_events[idx]
                _apply_reclose_line!(ps.network.ybus_real, tev)
                any_topology_change = true
                verbose && println("Reclose $idx: line $(tev.from_bus)-$(tev.to_bus) reclosed at t = $(tev.toff) s.")
            end
            any_event = true
            sched_idx += 1
        end

        # Refresh Y-preconditioner after topology change
        if any_topology_change && use_schur_gmres
            refresh_y_preconditioner!(gsw.prec, gsw.sw, ps, zold, dp.pvec, dt)
        end

        # re-solve at dt=0 after any event state change
        if any_event
            if use_schur
                newton_step_schur!(zold, f0, J0, sw, zold, dp.uvec, dp.pvec, ps, 0.0, verbose=verbose, tol=ftol, zwork=zwork, log=log, newton_norm=newton_norm)
            elseif use_schur_gmres
                newton_step_schur_gmres!(zold, f0, J0, gsw, zold, dp.uvec, dp.pvec, ps, 0.0, verbose=verbose, tol=ftol, zwork=zwork, log=log, newton_norm=newton_norm)
            else
                newton_step!(zold, f0, J0, fact, zold, dp.uvec, dp.pvec, ps, 0.0, verbose=verbose, jac_verify=false, tol=ftol, dx=dx_buf, zwork=zwork, log=log, newton_norm=newton_norm)
            end
        end

    end
    # ensure fault is deactivated at the end
    for event in events
        deactivate!(event)
    end

    dp.zvec = traj[:,end]
    return tvec, traj
end


function fill_pvec!(pvec::AbstractArray, dtype::AbstractDeviceType)
    @warn "fill_pvec! not implemented for device type $(dtype)"
end

function initial_guess!(
        xinit::AbstractArray,
        p::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::AbstractDeviceType
)
    @warn "initial_guess! not implemented for device type $(dtype)"
end

function initialize_dynamics!(
        f::AbstractArray,
        xinit::AbstractArray,
        p::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::AbstractDeviceType
)
    @warn "initialize_dynamics! not implemented for device type $(dtype)"
end

function preallocate_jacobian(ps::PowerSystem)
    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    ctrl_dim = ps.dynamic.ctrl_dim
    nbus = length(ps.buses)
    adj = ps.network.adjacency
    map = ps.dynamic.map

    # total system size
    sys_dim = diff_dim + alg_dim + 2*nbus

    # coordinate list for sparse jacobian
    coord_list = [Vector{Int}() for _ in 1:sys_dim]

    # diagonal non-zeros for integrators. After cluster-contiguous
    # reordering, diff equations are at psd.diff_indices positions
    # (no longer 1:diff_dim).
    di = ps.dynamic.diff_indices
    if di !== nothing
        for i in di
            push!(coord_list[i], i)
        end
    else
        for i in 1:diff_dim
            push!(coord_list[i], i)
        end
    end

    # network equations
    ptr = diff_dim + alg_dim + 1
    for fr in 1:nbus
        push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(fr - 1))
        push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(fr - 1) + 1)
        push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(fr - 1))
        push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(fr - 1) + 1)
        for to in adj[fr]
            push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(to - 1))
            push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(to - 1) + 1)
            push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(to - 1))
            push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(to - 1) + 1)
        end
    end

    # Per-device sparsity contribution (Genrou + ZIPLoad) — setup-time
    # only, runs once. Genrou and ZIPLoad each provide a
    # `preallocate_jacobian!(coord_list, ..., dtype)` method that pushes
    # their (row, col) entries into the coord_list. Uses device pointers
    # (updated by reorder_state!) not DynamicMap copies.
    for (i, device) in enumerate(ps.dynamic.devices)
        device.dtype isa IEESGO         && continue  # IEESGO sparsity comes from the batched preallocator below
        device.dtype isa TGOV1          && continue  # TGOV1 too
        device.dtype isa IEEEST         && continue  # IEEEST too
        device.dtype isa SEXS           && continue  # SEXS too
        device.dtype isa ESDC1A         && continue  # ESDC1A too
        device.dtype isa StaticGenerator && continue # StaticGenerator too
        bus = map.bus[i]
        diff_ptr = device.diff_ptr
        alg_ptr = diff_dim + device.alg_ptr
        volt_ptr = diff_dim + alg_dim + 2*(bus -1) + 1
        ctrl_ptr = device.ctrl_ptr
        preallocate_jacobian!(coord_list, diff_ptr, alg_ptr, ctrl_ptr, volt_ptr, device.dtype)
    end

    # IEESGO governor rows: no per-device `preallocate_jacobian!(::IEESGO)`
    # exists — its sparsity comes from the batched preallocator that
    # reads the SoA table directly.
    L = ps.dynamic.layout::SimulationLayout
    ieesgo_preallocate!(coord_list, L.ieesgo, diff_dim)
    tgov1_preallocate!(coord_list, L.tgov1, diff_dim)
    ieeest_preallocate!(coord_list, L.ieeest, diff_dim)
    sexs_preallocate!(coord_list, L.sexs)
    esdc1a_preallocate!(coord_list, L.esdc1a)
    static_gen_preallocate!(coord_list, L.static_gen)

    # Cross-device coupling sparsity: GENROU's swing eq reads governor p_m.
    # The legacy per-device GENROU preallocator can't see wiring; add it here.
    genrou_coupling_preallocate!(coord_list, L.genrou)

    # form coordinate lists (row, col, data)
    row = Int[]
    col = Int[]

    for i in 1:length(coord_list)
        if !isempty(coord_list[i])
            append!(row, fill(i, length(coord_list[i])))
            append!(col, coord_list[i])
        end
    end

    data = zeros(length(row))
    Jsp = sparse(row, col, data, sys_dim, sys_dim)

    # Now that J's sparsity is final, precompute every batched kernel's
    # nzval positions. These caches let the hot loop do
    # `nz[table.jac_pos[k, slot]] = val` without per-iteration row
    # search. Filled in once here per simulation setup.
    net_ptr = diff_dim + alg_dim
    genrou_jac_positions!(L.genrou, Jsp, diff_dim, net_ptr)
    ieesgo_jac_positions!(L.ieesgo, Jsp, diff_dim)
    tgov1_jac_positions!(L.tgov1, Jsp, diff_dim)
    ieeest_jac_positions!(L.ieeest, Jsp, diff_dim)
    sexs_jac_positions!(L.sexs, Jsp)
    esdc1a_jac_positions!(L.esdc1a, Jsp)
    zipload_jac_positions!(L.zipload, Jsp, net_ptr)
    static_gen_jac_positions!(L.static_gen, Jsp)

    return Jsp
end

function preallocate_jacobian!(
    coord_list::Vector{Vector{Int}},
    diff_ptr::Int,
    alg_ptr::Int,
    ctrl_ptr::Int,
    volt_ptr::Int,
    dtype::AbstractDeviceType
)
    error("preallocate_jacobian! not implemented for $(dtype) — either add a per-device sparsity contribution or route through a batched preallocator.")
end


function rhs_jac!(
    jac::AbstractMatrix,
    z::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    sys::PowerSystem)

    # Batched kernels are the sole hot path. See `rhs_fun!`
    # for the function-barrier rationale (Union field unboxing).
    dyn = sys.dynamic::PowerSystemDynamics
    net = sys.network::Network
    L = dyn.layout::SimulationLayout
    _rhs_jac_batched!(jac, z, u, p, dyn, net.ybus_real, L)
end

@noinline function _rhs_jac_batched!(jac::AbstractMatrix, z::AbstractArray, u::AbstractArray,
                                    p::AbstractArray, dyn::PowerSystemDynamics,
                                    ybus::SparseMatrixCSC, L::SimulationLayout)
    diff_dim = dyn.diff_dim
    alg_dim  = dyn.alg_dim
    net_ptr  = diff_dim + alg_dim

    current_injection_jacobian!(jac, ybus, net_ptr)
    genrou_jacobian_batch!(jac, z, u, p, L.genrou, diff_dim, net_ptr)
    ieesgo_jacobian_batch!(jac, p, L.ieesgo, diff_dim)
    tgov1_jacobian_batch!(jac, p, L.tgov1, diff_dim)
    ieeest_jacobian_batch!(jac, z, p, L.ieeest, diff_dim)
    sexs_jacobian_batch!(jac, z, p, L.sexs)
    esdc1a_jacobian_batch!(jac, z, p, L.esdc1a)
    zipload_jacobian_batch!(jac, z, p, L.zipload, net_ptr)
    static_gen_jacobian_batch!(jac, z, p, L.static_gen)

    _apply_events_jac!(jac, dyn.events, net_ptr)
    return nothing
end

@inline function _apply_events_jac!(jac::AbstractMatrix,
                                     events::Vector{ContingencyEvent}, net_ptr::Int)
    @inbounds for event in events
        if event.status
            bus = event.bus
            yfault = 1.0/event.rfault
            ptr1 = net_ptr+2*(bus-1)+1
            ptr2 = net_ptr+2*(bus-1)+2
            jac[ptr1, ptr1] += -yfault
            jac[ptr2, ptr2] += -yfault
        end
    end
    return nothing
end


"""
    current_injection_jacobian!(ybus::SparseMatrixCSC, jac::SparseMatrixCSC, dev::Int)

Copy the data from `ybus` to `jac` with an offset `dev` for row and column indices.
"""
function current_injection_jacobian!(jac::SparseMatrixCSC, ybus::SparseMatrixCSC, dev::Int)
    n_cols = size(ybus, 2)
    
    ybus_rows = rowvals(ybus)
    ybus_vals = nonzeros(ybus)
    
    jac_rows = rowvals(jac)
    jac_vals = nonzeros(jac)
    
    # Loop through each column
    for col_idx in 1:n_cols
        for i in nzrange(ybus, col_idx)
            row_idx = ybus_rows[i]
            val = -ybus_vals[i]
            
            # Calculate new row and column indices with offset dev
            new_row = row_idx + dev
            new_col = col_idx + dev
            
            # Find corresponding position in jac using nzrange
            for j in nzrange(jac, new_col)
                if jac_rows[j] == new_row
                    jac_vals[j] = val
                    break
                end
            end
        end
    end
end


export Genrou, ZIPLoad
export from_data_field
