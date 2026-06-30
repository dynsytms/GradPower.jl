# 2D scenario-indexed memory layout and batched CPU integration.
#
# BatchedLayout holds 2D arrays with scenario as the FIRST index for
# GPU coalescing in column-major Julia:
#   z[scenario, state], p[scenario, par], u[scenario, ctrl],
#   f[scenario, sys_dim], inj[scenario, 2*n_injectors],
#   J_nzval[scenario, nz_idx].
#
# The Jacobian sparsity pattern (rowval, colptr) is shared across all
# scenarios — only nzval varies per scenario.

# -----------------------------------------------------------------------
# BatchedLayout struct
# -----------------------------------------------------------------------

struct BatchedLayout
    M::Int                    # number of scenarios
    sys_dim::Int              # total system dimension per scenario
    diff_dim::Int
    alg_dim::Int
    nbus::Int
    z::Matrix{Float64}        # [M, sys_dim]
    p::Matrix{Float64}        # [M, par_dim]
    u::Matrix{Float64}        # [M, ctrl_dim]
    f::Matrix{Float64}        # [M, sys_dim]
    zold::Matrix{Float64}     # [M, sys_dim]
    inj::Matrix{Float64}      # [M, 2*n_injectors]
    J_nzval::Matrix{Float64}  # [M, nnz(J)]
    # Shared sparsity pattern (from single-scenario Jacobian)
    J_colptr::Vector{Int}
    J_rowval::Vector{Int}
    # Shared admittance matrix
    ybus::SparseMatrixCSC{Float64,Int}
    # Injection metadata
    inj_meta::InjectionMeta
    # uvec_idx routing (shared across scenarios)
    uvec_idx::Vector{Int64}
end

"""
    BatchedLayout(dp, ps, M)

Construct a BatchedLayout by replicating the single-scenario initial
condition across M scenarios. The Jacobian sparsity pattern is computed
once and shared.
"""
function BatchedLayout(dp::DynamicProblem, ps::PowerSystem, M::Int)
    dyn  = ps.dynamic::PowerSystemDynamics
    nbus = length(ps.buses)
    diff_dim = dyn.diff_dim
    alg_dim  = dyn.alg_dim
    sys_dim  = diff_dim + alg_dim + 2*nbus
    par_dim  = dyn.par_dim
    ctrl_dim = dyn.ctrl_dim
    L = dyn.layout::SimulationLayout

    @assert length(dp.zvec) == sys_dim
    @assert length(dp.pvec) == par_dim
    @assert length(dp.uvec) == ctrl_dim

    # Replicate z, p, u across M scenarios
    z    = zeros(Float64, M, sys_dim)
    p    = zeros(Float64, M, par_dim)
    u    = zeros(Float64, M, ctrl_dim)
    f    = zeros(Float64, M, sys_dim)
    zold = zeros(Float64, M, sys_dim)

    for m in 1:M
        z[m, :] .= dp.zvec
        p[m, :] .= dp.pvec
        u[m, :] .= dp.uvec
    end

    # Injection buffer
    inj_meta = InjectionMeta(L)
    inj = zeros(Float64, M, 2 * inj_meta.n_total)

    # Jacobian sparsity from single-scenario
    J_template = preallocate_jacobian(ps)
    nnz_J = length(J_template.nzval)
    J_nzval = zeros(Float64, M, nnz_J)
    J_colptr = copy(J_template.colptr)
    J_rowval = copy(J_template.rowval)

    ybus = ps.network.ybus_real
    uvec_idx = dyn.uvec_idx

    return BatchedLayout(M, sys_dim, diff_dim, alg_dim, nbus,
                         z, p, u, f, zold, inj,
                         J_nzval, J_colptr, J_rowval,
                         ybus, inj_meta, uvec_idx)
end

# -----------------------------------------------------------------------
# Batched helper functions
# -----------------------------------------------------------------------

"""
    _apply_uvec_routing_batched!(u, z, uvec_idx, M)

u[m, j] = z[m, uvec_idx[j]] for all scenarios m and all wired j.
"""
@inline function _apply_uvec_routing_batched!(u::Matrix{Float64}, z::Matrix{Float64},
                                               uvec_idx::Vector{Int64}, M::Int)
    @inbounds for j in eachindex(uvec_idx)
        src = uvec_idx[j]
        if src != 0
            for m in 1:M
                u[m, j] = z[m, src]
            end
        end
    end
    return nothing
end

"""
    _apply_events_fun_batched!(f, z, events, net_ptr, M)

Apply fault injection into f[m, :] for all scenarios m.
"""
@inline function _apply_events_fun_batched!(f::Matrix{Float64}, z::Matrix{Float64},
                                             events::Vector{ContingencyEvent},
                                             net_ptr::Int, M::Int)
    @inbounds for event in events
        if event.status
            bus = event.bus
            yfault = 1.0 / event.rfault
            vr_col = net_ptr + 2*(bus-1) + 1
            vi_col = vr_col + 1
            for m in 1:M
                f[m, vr_col] -= yfault * z[m, vr_col]
                f[m, vi_col] -= yfault * z[m, vi_col]
            end
        end
    end
    return nothing
end

"""
    _ybus_mul_batched!(f, z, ybus, net_ptr, M)

For each scenario m, compute f[m, net_ptr+1:end] = -ybus * z[m, net_ptr+1:end].
"""
function _ybus_mul_batched!(f::Matrix{Float64}, z::Matrix{Float64},
                             ybus::SparseMatrixCSC{Float64,Int},
                             net_ptr::Int, M::Int)
    nv = size(ybus, 1)
    rows = rowvals(ybus)
    vals = nonzeros(ybus)
    # Zero network portion of f
    @inbounds for m in 1:M
        for i in 1:nv
            f[m, net_ptr + i] = 0.0
        end
    end
    # Sparse matrix-vector multiply per scenario
    @inbounds for col in 1:nv
        for nz_idx in nzrange(ybus, col)
            row = rows[nz_idx]
            val = -vals[nz_idx]
            for m in 1:M
                f[m, net_ptr + row] += val * z[m, net_ptr + col]
            end
        end
    end
    return nothing
end

"""
    current_injection_jacobian_batched!(J_nzval, ybus, net_ptr, J_colptr, J_rowval, M)

Copy -ybus entries into J_nzval[m, :] for all scenarios m.
"""
function current_injection_jacobian_batched!(J_nzval::Matrix{Float64},
                                              ybus::SparseMatrixCSC{Float64,Int},
                                              net_ptr::Int,
                                              J_colptr::Vector{Int},
                                              J_rowval::Vector{Int},
                                              M::Int)
    ybus_rows = rowvals(ybus)
    ybus_vals = nonzeros(ybus)
    n_cols = size(ybus, 2)
    @inbounds for col_idx in 1:n_cols
        for i in nzrange(ybus, col_idx)
            row_idx = ybus_rows[i]
            val = -ybus_vals[i]
            new_row = row_idx + net_ptr
            new_col = col_idx + net_ptr
            # Find position in J_nzval
            for j in J_colptr[new_col]:(J_colptr[new_col+1]-1)
                if J_rowval[j] == new_row
                    for m in 1:M
                        J_nzval[m, j] = val
                    end
                    break
                end
            end
        end
    end
    return nothing
end

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
# Per-scenario residual and Jacobian using KA CPU path (single scenario
# from 2D layout, mapped to 1D views for the existing kernels)
# -----------------------------------------------------------------------

"""
    _residual_scenario!(f_row, z_row, u_row, p_row, dyn, ybus, L)

Evaluate the residual for a single scenario using the plain-loop path.
"""
function _residual_scenario!(f_row::AbstractVector, z_row::AbstractVector,
                              u_row::AbstractVector, p_row::AbstractVector,
                              dyn::PowerSystemDynamics,
                              ybus::SparseMatrixCSC, L::SimulationLayout)
    _rhs_fun_batched!(f_row, z_row, u_row, p_row, dyn, ybus, L)
    return nothing
end

"""
    _jacobian_scenario!(J_nzval_row, z_row, u_row, p_row, dyn, ybus, L, J_colptr, J_rowval, sys_dim)

Evaluate the Jacobian nzval for a single scenario.
"""
function _jacobian_scenario!(J_nzval_row::AbstractVector, z_row::AbstractVector,
                              u_row::AbstractVector, p_row::AbstractVector,
                              dyn::PowerSystemDynamics,
                              ybus::SparseMatrixCSC, L::SimulationLayout,
                              J_colptr::Vector{Int}, J_rowval::Vector{Int},
                              sys_dim::Int, J_buf::SparseMatrixCSC)
    # Reuse pre-built J_buf: copy nzval pointer content, fill, evaluate,
    # then copy results back to J_nzval_row.
    fill!(J_buf.nzval, 0.0)
    _rhs_jac_batched!(J_buf, z_row, u_row, p_row, dyn, ybus, L)
    copyto!(J_nzval_row, J_buf.nzval)
    return nothing
end

# -----------------------------------------------------------------------
# Batched residual evaluation (all M scenarios)
# -----------------------------------------------------------------------

function _rhs_fun_all_scenarios!(bl::BatchedLayout, dyn::PowerSystemDynamics,
                                  L::SimulationLayout)
    M = bl.M
    for m in 1:M
        f_row   = @view bl.f[m, :]
        z_row   = @view bl.z[m, :]
        u_row   = @view bl.u[m, :]
        p_row   = @view bl.p[m, :]
        _residual_scenario!(f_row, z_row, u_row, p_row,
                            dyn, bl.ybus, L)
    end
    return nothing
end

# -----------------------------------------------------------------------
# Batched Jacobian evaluation (all M scenarios)
# -----------------------------------------------------------------------

function _rhs_jac_all_scenarios!(bl::BatchedLayout, dyn::PowerSystemDynamics,
                                  L::SimulationLayout)
    M = bl.M
    # Build one reusable SparseMatrixCSC with its own nzval buffer
    nnz_J = size(bl.J_nzval, 2)
    J_buf = SparseMatrixCSC(bl.sys_dim, bl.sys_dim,
                            bl.J_colptr, bl.J_rowval,
                            zeros(Float64, nnz_J))
    for m in 1:M
        nzv     = @view bl.J_nzval[m, :]
        z_row   = @view bl.z[m, :]
        u_row   = @view bl.u[m, :]
        p_row   = @view bl.p[m, :]
        _jacobian_scenario!(nzv, z_row, u_row, p_row,
                            dyn, bl.ybus, L,
                            bl.J_colptr, bl.J_rowval, bl.sys_dim, J_buf)
    end
    return nothing
end

# -----------------------------------------------------------------------
# Backward Euler for all scenarios
# -----------------------------------------------------------------------

function _beuler_all_scenarios!(bl::BatchedLayout, dyn::PowerSystemDynamics,
                                 L::SimulationLayout, dt::Float64)
    _rhs_fun_all_scenarios!(bl, dyn, L)
    beuler_batched_2d!(bl.f, bl.z, bl.zold,
                       dyn.is_diff, dyn.diff_indices,
                       bl.diff_dim, dt, bl.M)
    return nothing
end

function _beuler_jac_all_scenarios!(bl::BatchedLayout, dyn::PowerSystemDynamics,
                                     L::SimulationLayout, dt::Float64)
    _rhs_jac_all_scenarios!(bl, dyn, L)
    # Apply backward Euler scaling per scenario
    is_diff = dyn.is_diff
    n_da = is_diff !== nothing ? length(is_diff) : bl.diff_dim
    @inbounds for m in 1:bl.M
        nzv = @view bl.J_nzval[m, :]
        # Scale diff rows: nz *= -dt, diagonal += 1
        for col in 1:bl.sys_dim
            for nz_idx in bl.J_colptr[col]:(bl.J_colptr[col+1]-1)
                row = bl.J_rowval[nz_idx]
                if is_diff !== nothing
                    if row <= n_da && is_diff[row]
                        nzv[nz_idx] *= -dt
                        if row == col
                            nzv[nz_idx] += 1.0
                        end
                    end
                else
                    if row <= bl.diff_dim
                        nzv[nz_idx] *= -dt
                        if row == col
                            nzv[nz_idx] += 1.0
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# -----------------------------------------------------------------------
# Batched Newton step (per-scenario sequential KLU)
# -----------------------------------------------------------------------

function _newton_step_batched!(bl::BatchedLayout, dyn::PowerSystemDynamics,
                                L::SimulationLayout, dt::Float64,
                                J_bufs::Vector{SparseMatrixCSC{Float64,Int}},
                                facts::Vector{Any},
                                dx_buf::Vector{Float64},
                                f_buf::Vector{Float64};
                                itermax::Int=30, tol::Float64=1e-10)
    M = bl.M
    sys_dim = bl.sys_dim

    for iter in 1:itermax
        # Evaluate residual
        _beuler_all_scenarios!(bl, dyn, L, dt)

        # Check convergence (max over all scenarios)
        max_norm = 0.0
        @inbounds for m in 1:M
            for i in 1:sys_dim
                a = abs(bl.f[m, i])
                if a > max_norm
                    max_norm = a
                end
            end
        end
        max_norm < tol && return true

        # Evaluate Jacobian
        _beuler_jac_all_scenarios!(bl, dyn, L, dt)

        # Solve per scenario — match newton_step! pattern: klu() on iter 1, klu!() after
        for m in 1:M
            J_m = J_bufs[m]
            copyto!(J_m.nzval, @view bl.J_nzval[m, :])
            copyto!(f_buf, @view bl.f[m, :])
            if iter == 1
                facts[m] = klu(J_m)
            else
                klu!(facts[m], J_m)
            end
            ldiv!(dx_buf, facts[m], f_buf)
            @inbounds for k in 1:sys_dim
                bl.z[m, k] -= dx_buf[k]
            end
        end
    end
    return false
end

# -----------------------------------------------------------------------
# integrate_batched!
# -----------------------------------------------------------------------

"""
    integrate_batched!(bl, ps, tf; dt=1/120)

Batched CPU integration using the KA CPU backend with the 2D layout.
Returns (tvec, traj) where traj[m] is the trajectory for scenario m.
"""
function integrate_batched!(bl::BatchedLayout, ps::PowerSystem, tf::Float64;
                             dt::Float64=1.0/120.0, newton_tol::Float64=1e-10)
    dyn = ps.dynamic::PowerSystemDynamics
    L = dyn.layout::SimulationLayout
    M = bl.M
    sys_dim = bl.sys_dim

    nsteps = Int(round(tf / dt))
    tvec = collect(0:dt:tf)

    # Build event schedule
    events = dyn.events
    event_schedule = Tuple{Int,Int,Symbol}[]
    for (ei, ev) in enumerate(events)
        push!(event_schedule, (Int(round(ev.ton / dt)), ei, :on))
        push!(event_schedule, (Int(round(ev.toff / dt)), ei, :off))
    end
    sort!(event_schedule, by = x -> x[1])

    # Allocate per-scenario trajectories
    trajs = [zeros(Float64, sys_dim, nsteps + 1) for _ in 1:M]
    for m in 1:M
        trajs[m][:, 1] .= @view bl.z[m, :]
    end

    # Copy initial z to zold
    copyto!(bl.zold, bl.z)

    # Pre-allocate per-scenario J buffers and scratch vectors
    nnz_J = size(bl.J_nzval, 2)
    J_bufs = [SparseMatrixCSC(sys_dim, sys_dim, copy(bl.J_colptr), copy(bl.J_rowval),
                               zeros(Float64, nnz_J)) for _ in 1:M]
    facts = Vector{Any}(undef, M)
    dx_buf = zeros(Float64, sys_dim)
    f_buf  = zeros(Float64, sys_dim)

    sched_idx = 1
    for k in 1:nsteps
        # Copy current z to zold
        copyto!(bl.zold, bl.z)

        # Newton step
        _newton_step_batched!(bl, dyn, L, dt, J_bufs, facts, dx_buf, f_buf; tol=newton_tol)

        # Store trajectory
        for m in 1:M
            trajs[m][:, k + 1] .= @view bl.z[m, :]
        end

        # Process events
        any_event = false
        while sched_idx <= length(event_schedule) && event_schedule[sched_idx][1] == k
            _, idx, action = event_schedule[sched_idx]
            if action === :on
                activate!(events[idx])
            elseif action === :off
                deactivate!(events[idx])
            end
            any_event = true
            sched_idx += 1
        end

        if any_event
            # Re-solve at dt=0 after event
            copyto!(bl.zold, bl.z)
            _newton_step_batched!(bl, dyn, L, 0.0, J_bufs, facts, dx_buf, f_buf; tol=newton_tol)
        end
    end

    # Deactivate events
    for event in events
        deactivate!(event)
    end

    return tvec, trajs
end
