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

    # Initialize ZIPLoad devices with power-flow voltage magnitudes.
    for (i, device) in enumerate(ps.dynamic.devices)
        if device.dtype isa ZIPLoad
            device.dtype.v0mag = ps.buses[ps.dynamic.map.bus[i]].v0m
            yload = (device.dtype.pinj + 1im*device.dtype.qinj)/(device.dtype.v0mag^2.0)
            device.dtype.yreal = real(yload)
            device.dtype.yimag = imag(yload)
        end
    end
    
    # Fill parameter vector.
    # NOTE: ZIPLoad parameters depend on the power-flow solution (v0mag, yreal, yimag),
    # so pvec must be filled after the ZIPLoad initialization above.
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
        # Re-read the regulated setpoints from the power-flow solution rather
        # than trusting the values captured at parse time. Without Q-limit
        # enforcement these are identical -- a PV bus's magnitude is not a
        # power-flow variable, so `v0m` never leaves `vs`. With enforcement, a
        # bus that was converted to PQ has a solved voltage that differs from
        # its original setpoint, and a stub still holding the old `vset` makes
        # `vr^2 + vi^2 = vset^2` inconsistent with the operating point (0.08 pu
        # of initialization residual on ACTIVSg2000).
        sg.vset = ps.buses[map.bus[i]].v0m
        sg.aset = ps.buses[map.bus[i]].v0a
        # A PV stub whose bus the power flow converted to PQ is at a reactive
        # bound and can no longer hold its voltage. Its alg state stays (the
        # layout was fixed before the power flow ran) but switches from the
        # voltage-regulation residual to `q = q0`. Without this the stub keeps
        # regulating with unlimited Q through the transient, which is exactly
        # the behaviour Q-limit enforcement was meant to remove.
        sg.regulating = sg.bus_type != 2 || ps.buses[map.bus[i]].type == 2
        # Write alg states using absolute z-index (diff_dim + alg_ptr).
        alg_global = diff_dim + device.alg_ptr
        if sg.bus_type == 2          # PV: alg state is q
            z[alg_global] = qsum
        elseif sg.bus_type == 3      # SLACK: alg states are p, q
            z[alg_global]     = psum
            z[alg_global + 1] = qsum
        end
        # mirror p0, q0, vset, aset into pvec so the kernel reads correct
        # values. pvec was filled above, before these were refreshed.
        p[device.par_ptr]     = psum
        p[device.par_ptr + 1] = qsum
        p[device.par_ptr + 2] = sg.vset
        p[device.par_ptr + 3] = sg.aset
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

    _zero_offline_rows!(f, dyn)
    _apply_events_fun!(f, v, dyn.events, net_ptr)
    return nothing
end

# Offline devices are skipped by the kernels, so their rows of `f` would keep
# stale values. Zero them: diff rows then become `z - zold = 0` under backward
# Euler and alg rows `0 = 0` (with the unit diagonal set by
# `_pin_offline_jacobian!`), i.e. the states are frozen.
@inline function _zero_offline_rows!(f::AbstractArray, dyn::PowerSystemDynamics)
    @inbounds for i in dyn.offline_diff_rows
        f[i] = 0.0
    end
    @inbounds for i in dyn.offline_alg_rows
        f[i] = 0.0
    end
    return nothing
end

# Unit diagonal on offline alg rows (their other entries are zero because the
# kernels skip offline devices). `preallocate_jacobian` puts the diagonal in
# the pattern for every device with a scheduled disconnect.
function _pin_offline_jacobian!(jac::SparseMatrixCSC, dyn::PowerSystemDynamics)
    rv = rowvals(jac)
    nz = nonzeros(jac)
    @inbounds for i in dyn.offline_alg_rows
        r = nzrange(jac, i)
        j = first(r) + searchsortedfirst(view(rv, r), i) - 1
        (j <= last(r) && rv[j] == i) ||
            error("offline alg row $i has no structural diagonal; disconnect events must be added before preallocate_jacobian")
        nz[j] = 1.0
    end
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

# Sum-of-squares speed objective. Extend via multiple dispatch for other objectives.
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
    dt::Float64,
    limits::Union{Nothing,LimitWorkspace}=nothing,
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
    limits === nothing || apply_limit_residual!(f, z, zold, p, dt, limits)
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
    limits::Union{Nothing,LimitWorkspace}=nothing,
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
    limits === nothing || apply_limit_residual!(f, z, zold, p, dt, limits)
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
    dt::Float64,
    limits::Union{Nothing,LimitWorkspace}=nothing,
)
    # set all elements to zero
    fill!(J.nzval, 0.0)

    # evaluate Jacobian
    rhs_jac!(J, z, u, p, sys)

    # scale for backward Euler
    _jacobian_beuler_positions!(J, sys.dynamic.layout::SimulationLayout, dt)
    limits === nothing || apply_limit_jacobian!(J, z, zold, p, dt, limits)
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
    zold::Union{Nothing,AbstractVector}=nothing,
    limits::Union{Nothing,LimitWorkspace}=nothing,
)
    fill!(J.nzval, 0.0)
    _rhs_jac_batched!(J, z, u, p, dyn, ybus, L)
    _jacobian_beuler_positions!(J, L, dt)
    if limits !== nothing
        zold === nothing && error("limited backward-Euler Jacobian requires zold")
        apply_limit_jacobian!(J, z, zold, p, dt, limits)
    end
    return nothing
end

"""
    beuler_jac_positions(J, is_diff) -> (row_pos, diag_pos)

Indices into `nonzeros(J)` of every entry in a differential row
(`is_diff[row]`) and of each differential row's diagonal. Errors if a
differential row has no structural diagonal.
"""
function beuler_jac_positions(J::SparseMatrixCSC, is_diff::BitVector)
    rv = rowvals(J)
    row_pos = Int32[]
    diag_pos = Int32[]
    n_da = length(is_diff)
    for col in 1:size(J, 2), k in nzrange(J, col)
        row = rv[k]
        if row <= n_da && is_diff[row]
            push!(row_pos, k)
            row == col && push!(diag_pos, k)
        end
    end
    length(diag_pos) == count(is_diff) ||
        error("Jacobian pattern is missing a differential-row diagonal")
    return row_pos, diag_pos
end

"""
    _jacobian_beuler_positions!(J, L, h)

Backward-Euler Jacobian of `dz/dt = f`: on differential rows,
`J[i,:] *= -h` and `J[i,i] += 1`, through the positions cached by
`preallocate_jacobian`.
"""
function _jacobian_beuler_positions!(J::SparseMatrixCSC, L::SimulationLayout, h::Float64)
    nz = nonzeros(J)
    @inbounds for k in L.diff_row_pos
        nz[k] *= -h
    end
    @inbounds for k in L.diff_diag_pos
        nz[k] += 1.0
    end
    return nothing
end

"""
    integrate!(dp, ps, tf; dt=1/120, solver=:monolithic, linear_solver=KLUBackend(),
               newton_tol=1e-10, jacobian_reuse=true, predictor=true, ...) -> (tvec, traj)

Backward-Euler integration of the DAE from `dp.zvec` to `tf`.

`jacobian_reuse=true` keeps the Jacobian factorization across Newton
iterations and time steps (modified Newton) and refactors only when
convergence slows down, at events, or after a failed solve; see
[`JacobianReuse`](@ref). `predictor=true` starts each step from the linear
extrapolation `2 z_k - z_{k-1}` (not across events). Both converge to the same
`newton_tol`; set both to `false` for full Newton from the previous state.
They apply to `solver=:monolithic` and `:schur`; `:schur_gmres` and
`limit_method != :none` always use full Newton.

# Output

By default the full trajectory is returned: `traj[:, j]` is the state at
`tvec[j]`, for every step. To store less:

  * `save_every = s` stores the steps `1:s:nsteps+1` (step 1 = initial state);
    the final step is not added when it is not on that grid. `tvec` holds the
    stored times.
  * `save_idxs` stores only those rows of the state (`traj[r, j] =
    z[save_idxs[r]]`); default `:` (all rows).
  * `save_eltype = Float32` stores in that element type.
  * `step_callback(k, t, z)` is called after the initial state (`k = 1`) and
    after every accepted step `k = 2, …, nsteps+1`, with the full state `z` at
    time `t`, whether or not it is stored. `z` must not be modified.

At an event step the stored (and passed) state is the one before the event's
algebraic re-solve. These options do not change the computed solution;
`dp.zvec` is always set to the final state.
"""
function integrate!(
    dp::DynamicProblem,
    ps::PowerSystem,
    tf::Float64;
    dt::Float64=(1.0/120.0),
    verbose::Bool=false,
    log::Union{Nothing,SolverLog}=nothing,
    solver::Symbol=:monolithic,
    linear_solver::AbstractDirectSolverBackend=KLUBackend(),
    newton_tol::Float64=1e-10,
    newton_norm::Symbol=:inf,
    limit_method::Symbol=:none,
    limit_mu::Float64=0.0,
    limit_tolerance::Float64=1e-10,
    limit_events::Union{Nothing,Vector{LimitEvent}}=nothing,
    jacobian_reuse::Bool=true,
    predictor::Bool=true,
    save_every::Integer=1,
    save_idxs=Colon(),
    save_eltype::Type{<:AbstractFloat}=Float64,
    step_callback=nothing,
)
    # NOTE: Assumes initialize_dynamics!(dp, ps) has been called.

    # calculate number of steps
    nsteps, tvec = _time_grid(tf, dt)

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

    # Stored output (see "Output" in the docstring); `zlast` is the previous
    # accepted state, needed by the limit-event bookkeeping.
    rec = _Recorder(system_size, nsteps, save_idxs, save_every, save_eltype)
    zlast = limit_method === :none ? nothing : similar(dp.zvec, Float64)

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
    _record!(rec, 1, zold)
    step_callback === nothing || step_callback(1, tvec[1], zold)
    zlast === nothing || copyto!(zlast, zold)
    zfinal = copy(zold)   # state at the last grid step (see end of function)
    limit_method !== :none && solver !== :monolithic &&
        throw(ArgumentError("dynamic limits currently support solver=:monolithic only"))

    # initialize residual and Jacobian
    f0 = zeros(Float64, system_size)
    J0 = preallocate_jacobian(ps)
    limit_workspace = build_limit_workspace(ps, J0, zold, dp.pvec;
        method=limit_method, mu=limit_mu, tolerance=limit_tolerance,
        events=limit_events === nothing ? LimitEvent[] : limit_events)
    beuler_jac!(J0, zold, zold, dp.uvec, dp.pvec, ps, ps.dynamic.diff_dim, dt,
                limit_workspace)

    # pre-factorization (symbolic analysis reused by every Newton iteration)
    fact = Ref{Any}(ds_solver(linear_solver, J0; tuned=true))

    # Modified-Newton state shared by time steps and dt = 0 event re-solves
    # (both use J0/fact; a change of dt forces a refactorization). Limits
    # change Jacobian rows with the active set and GMRES has its own
    # preconditioner reuse, so both keep full Newton.
    reuse = JacobianReuse(jacobian_reuse && limit_method === :none && solver !== :schur_gmres)
    use_predictor = predictor && limit_method === :none
    zprev = similar(zold)   # z_{k-1}, for the predictor
    zpred = similar(zold)   # predicted initial guess for step k
    have_prev = false

    # Schur workspace (preallocated once, reused every step)
    use_schur = solver === :schur
    use_schur_gmres = solver === :schur_gmres
    sw = use_schur ? SchurWorkspace(ps; linear_solver) : nothing
    gsw = use_schur_gmres ? GmresSchurWorkspace(ps, zold, dp.pvec, dt; linear_solver) : nothing

    # Events mutate the system (Ybus for trips, online flags for
    # disconnects). Snapshot both and restore them on exit, including on
    # error, so repeated integrations of the same `ps` are independent.
    saved = _snapshot_event_state(ps)

    try
    # time loop
    sched_idx = 1  # pointer into sorted event_schedule
    for k in 1:nsteps
        verbose && println("Time-stepping. t = $(tvec[k]) s.")
        # Initial guess: linear extrapolation from the last two states.
        # `zold` stays the backward-Euler anchor z_k.
        zguess = zold
        if use_predictor && have_prev
            zguess = zpred
            @inbounds for i in eachindex(zguess)
                zguess[i] = 2.0*zold[i] - zprev[i]
            end
        end
        use_predictor && copyto!(zprev, zold)
        if use_schur
            success = newton_step_schur!(zguess, f0, J0, sw, zold, dp.uvec, dp.pvec, ps, dt, verbose=verbose, tol=ftol, zwork=zwork, log=log, newton_norm=newton_norm, reuse=reuse)
        elseif use_schur_gmres
            success = newton_step_schur_gmres!(zguess, f0, J0, gsw, zold, dp.uvec, dp.pvec, ps, dt, verbose=verbose, tol=ftol, zwork=zwork, log=log, newton_norm=newton_norm)
        else
            success = newton_step!(zguess, f0, J0, fact, zold, dp.uvec, dp.pvec, ps, dt, verbose=verbose, jac_verify=false, tol=ftol, dx=dx_buf, zwork=zwork, log=log, newton_norm=newton_norm, limits=limit_workspace, reuse=reuse)
        end
        check_newton(success, tvec[k+1], "time step (solver=$solver, limit_method=$limit_method)")
        zguess === zold || copyto!(zold, zguess)
        have_prev = true
        if zlast !== nothing
            record_limit_events!(limit_workspace, zold, zlast, dp.pvec, dt, tvec[k+1])
            copyto!(zlast, zold)
        end
        _record!(rec, k + 1, zold)
        step_callback === nothing || step_callback(k + 1, tvec[k+1], zold)
        k == nsteps && copyto!(zfinal, zold)

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
            # The event changed the equations: the cached factorization is
            # stale, and the trajectory is not smooth across the event.
            reuse.dt = NaN
            have_prev = false
            if use_schur
                success = newton_step_schur!(zold, f0, J0, sw, zold, dp.uvec, dp.pvec, ps, 0.0, verbose=verbose, tol=ftol, zwork=zwork, log=log, newton_norm=newton_norm, reuse=reuse)
            elseif use_schur_gmres
                success = newton_step_schur_gmres!(zold, f0, J0, gsw, zold, dp.uvec, dp.pvec, ps, 0.0, verbose=verbose, tol=ftol, zwork=zwork, log=log, newton_norm=newton_norm)
            else
                success = newton_step!(zold, f0, J0, fact, zold, dp.uvec, dp.pvec, ps, 0.0, verbose=verbose, jac_verify=false, tol=ftol, dx=dx_buf, zwork=zwork, log=log, newton_norm=newton_norm, limits=limit_workspace, reuse=reuse)
            end
            check_newton(success, tvec[k+1], "dt=0 event re-solve (solver=$solver, limit_method=$limit_method)")
        end

    end
    finally
        _restore_event_state!(ps, saved)
    end

    # Final state: the last grid step, before any event re-solve at that step
    # (the convention of the stored trajectory, and the previous behavior).
    dp.zvec = zfinal
    return _saved_times(tvec, rec.every), rec.traj
end

function _snapshot_event_state(ps::PowerSystem)
    dyn = ps.dynamic::PowerSystemDynamics
    L = dyn.layout::SimulationLayout
    return (ybus_nzval = copy(nonzeros(ps.network.ybus_real)),
            device_online = [d.online for d in dyn.devices],
            offline_rows = (copy(dyn.offline_diff_rows), copy(dyn.offline_alg_rows)),
            table_online = map(t -> copy(t.online), values(L.tables)))
end

function _restore_event_state!(ps::PowerSystem, saved)
    dyn = ps.dynamic::PowerSystemDynamics
    L = dyn.layout::SimulationLayout
    # Trips/recloses only touch existing entries, so the sparsity pattern is
    # unchanged and restoring nzval is exact.
    copyto!(nonzeros(ps.network.ybus_real), saved.ybus_nzval)
    for (d, on) in zip(dyn.devices, saved.device_online)
        d.online = on
    end
    foreach((t, on) -> copyto!(t.online, on), values(L.tables), saved.table_online)
    copy!(dyn.offline_diff_rows, saved.offline_rows[1])
    copy!(dyn.offline_alg_rows, saved.offline_rows[2])
    foreach(deactivate!, dyn.events)
    return nothing
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

    # Alg rows of devices scheduled for disconnection get a structural
    # diagonal: some have none (e.g. Genrou's v_q/v_d/i_q/i_d), and an
    # offline device's alg rows are pinned to the identity
    # (`_pin_offline_jacobian!`). Duplicates are summed away by `sparse`.
    for ev in ps.dynamic.disconnect_events
        dev = ps.dynamic.devices[ev.device_idx]
        a0 = diff_dim + dev.alg_ptr
        for i in a0:a0+dev.dtype.alg_size-1
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
    copy!(L.net_jac_pos, ybus_jac_positions(ps.network.ybus_real, Jsp, net_ptr))
    row_pos, diag_pos = beuler_jac_positions(Jsp, ps.dynamic.is_diff)
    copy!(L.diff_row_pos, row_pos)
    copy!(L.diff_diag_pos, diag_pos)
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

    current_injection_jacobian!(jac, ybus, L.net_jac_pos)
    genrou_jacobian_batch!(jac, z, u, p, L.genrou, diff_dim, net_ptr)
    ieesgo_jacobian_batch!(jac, p, L.ieesgo, diff_dim)
    tgov1_jacobian_batch!(jac, p, L.tgov1, diff_dim)
    ieeest_jacobian_batch!(jac, z, p, L.ieeest, diff_dim)
    sexs_jacobian_batch!(jac, z, p, L.sexs)
    esdc1a_jacobian_batch!(jac, z, p, L.esdc1a)
    zipload_jacobian_batch!(jac, z, p, L.zipload, net_ptr)
    static_gen_jacobian_batch!(jac, z, p, L.static_gen)

    isempty(dyn.offline_alg_rows) || _pin_offline_jacobian!(jac, dyn)
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
    ybus_jac_positions(ybus, J, net_ptr) -> Vector{Int32}

For each stored entry `k` of `ybus`, the index into `nonzeros(J)` of the
entry at `(row + net_ptr, col + net_ptr)`. Errors if `J`'s pattern does not
contain the whole network block.
"""
function ybus_jac_positions(ybus::SparseMatrixCSC, J::SparseMatrixCSC, net_ptr::Int)
    yrows = rowvals(ybus)
    jrows = rowvals(J)
    pos = Vector{Int32}(undef, nnz(ybus))
    for col in 1:size(ybus, 2)
        jr = nzrange(J, col + net_ptr)
        for k in nzrange(ybus, col)
            row = yrows[k] + net_ptr
            j = first(jr) + searchsortedfirst(view(jrows, jr), row) - 1
            (j <= last(jr) && jrows[j] == row) ||
                error("Jacobian pattern is missing network entry ($row, $(col + net_ptr))")
            pos[k] = j
        end
    end
    return pos
end

"""
    current_injection_jacobian!(jac, ybus, pos)

Write `-ybus` into the network block of `jac`, using the positions
`pos = ybus_jac_positions(ybus, jac, net_ptr)`. The Ybus sparsity pattern
must be the one `pos` was built for (line trips only change values).
"""
function current_injection_jacobian!(jac::SparseMatrixCSC, ybus::SparseMatrixCSC,
                                     pos::Vector{Int32})
    yv = nonzeros(ybus)
    jv = nonzeros(jac)
    length(pos) == length(yv) ||
        error("stale Ybus -> Jacobian positions: rebuild with preallocate_jacobian")
    @inbounds for k in eachindex(pos)
        jv[pos[k]] = -yv[k]
    end
    return nothing
end


export Genrou, ZIPLoad
export from_data_field
