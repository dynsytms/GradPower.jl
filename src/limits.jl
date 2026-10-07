const LIMIT_FREE  = UInt8(0)
const LIMIT_LOWER = UInt8(1)
const LIMIT_UPPER = UInt8(2)

function build_limit_workspace(ps::PowerSystem, J::SparseMatrixCSC, z, p;
                               method::Symbol, mu::Float64,
                               tolerance::Float64,
                               events::Vector{LimitEvent}=LimitEvent[])
    method in (:none, :active_set, :complementarity) ||
        throw(ArgumentError("limit_method must be :none, :active_set, or :complementarity"))
    mu >= 0.0 || throw(ArgumentError("limit_mu must be non-negative"))
    tolerance >= 0.0 || throw(ArgumentError("limit_tolerance must be non-negative"))

    dyn = ps.dynamic::PowerSystemDynamics
    descriptors = FixedStateLimit[]
    if method !== :none
        table = dyn.layout.tgov1
        device_ids = String[]
        for device in dyn.devices
            device.dtype isa TGOV1 || continue
            push!(device_ids, _normalize_id(device.dtype.id))
        end
        for k in 1:table.n
            table.online[k] || continue
            dp = Int(table.diff_ptr[k])
            pp = Int(table.par_ptr[k])
            push!(descriptors, FixedStateLimit(
                k, dp + 1, Int(table.w_idx[k]),
                Int(table.jac_pos[k, J_TG_R2_x2]), Int(table.jac_pos[k, J_TG_R2_w]),
                pp + 3, pp + 2, :TGOV1,
                Int(ps.buses[Int(table.bus[k])].i), device_ids[k],
            ))
        end
    end

    workspace = LimitWorkspace(method, mu, tolerance, descriptors,
                               method === :none ? Bool[] : dyn.layout.tgov1.online,
                               fill(LIMIT_FREE, length(descriptors)), events)
    validate_initial_limits!(workspace, z, p)
    return workspace
end

function validate_initial_limits!(workspace::LimitWorkspace, z, p)
    for descriptor in workspace.descriptors
        lower = p[descriptor.lower_parameter_index]
        upper = p[descriptor.upper_parameter_index]
        isfinite(lower) && isfinite(upper) ||
            throw(ArgumentError("$(descriptor.device_type) limits must be finite"))
        lower < upper ||
            throw(ArgumentError("$(descriptor.device_type) lower limit must be less than upper limit"))
        value = z[descriptor.state_index]
        lower - workspace.tolerance <= value <= upper + workspace.tolerance ||
            throw(ArgumentError("$(descriptor.device_type) initial state $value is outside [$lower, $upper]"))
    end
    return nothing
end

@inline function _fb_value(a::Float64, b::Float64, mu::Float64)
    return sqrt(a*a + b*b + mu*mu) - a - b
end

@inline function _fb_derivatives(a::Float64, b::Float64, mu::Float64)
    rho = sqrt(a*a + b*b + mu*mu)
    rho == 0.0 && return (-1.0, -1.0)
    return (a/rho - 1.0, b/rho - 1.0)
end

@inline function _apply_limit_residual!(f, z, p, dt::Float64,
                                        workspace::LimitWorkspace)
    method = workspace.method
    method === :none && return nothing
    @inbounds for descriptor in workspace.descriptors
        workspace.online[descriptor.table_index] || continue
        row = descriptor.state_index
        x = z[row]
        lower = p[descriptor.lower_parameter_index]
        upper = p[descriptor.upper_parameter_index]
        free = f[row]
        if method === :active_set
            projected = clamp(x - free, lower, upper)
            f[row] = x - projected
        else
            mu = dt == 0.0 ? 0.0 : workspace.mu
            inner = _fb_value(upper - x, -free, mu)
            f[row] = _fb_value(x - lower, inner, mu)
        end
    end
    return nothing
end

function _transform_limit_jacobian!(J::SparseMatrixCSC, z, zold, p,
                                    dt::Float64, workspace::LimitWorkspace)
    method = workspace.method
    method === :none && return nothing
    @inbounds for descriptor in workspace.descriptors
        workspace.online[descriptor.table_index] || continue
        row = descriptor.state_index
        x = z[row]
        lower = p[descriptor.lower_parameter_index]
        upper = p[descriptor.upper_parameter_index]
        free = _free_limit_residual(descriptor, z, zold, p, dt)

        if method === :active_set
            y = x - free
            if y <= lower || y >= upper
                J.nzval[descriptor.diagonal_position] = 1.0
                descriptor.input_position == 0 ||
                    (J.nzval[descriptor.input_position] = 0.0)
            end
        else
            mu = dt == 0.0 ? 0.0 : workspace.mu
            inner = _fb_value(upper - x, -free, mu)
            outer_a, outer_b = _fb_derivatives(x - lower, inner, mu)
            inner_c, inner_q = _fb_derivatives(upper - x, -free, mu)
            scale = -outer_b * inner_q
            diagonal = outer_a - outer_b * inner_c
            J.nzval[descriptor.diagonal_position] =
                scale * J.nzval[descriptor.diagonal_position] + diagonal
            descriptor.input_position == 0 ||
                (J.nzval[descriptor.input_position] *= scale)
        end
    end
    return nothing
end

function _free_limit_residual(descriptor::FixedStateLimit, z, zold, p,
                              dt::Float64)
    descriptor.device_type === :TGOV1 || error("unsupported fixed-state limit")
    pp = descriptor.upper_parameter_index - 2
    R = p[pp]
    T1 = p[pp + 1]
    pref = p[pp + 7]
    x = z[descriptor.state_index]
    w = descriptor.input_index == 0 ? 0.0 : z[descriptor.input_index]
    return x - zold[descriptor.state_index] - dt * (((pref - w)/R - x)/T1)
end

function apply_limit_residual!(f, z, zold, p, dt, workspace::LimitWorkspace)
    _apply_limit_residual!(f, z, p, dt, workspace)
    return nothing
end

function apply_limit_jacobian!(J, z, zold, p, dt, workspace::LimitWorkspace)
    _transform_limit_jacobian!(J, z, zold, p, dt, workspace)
    return nothing
end

function record_limit_events!(workspace::LimitWorkspace, z, zold, p,
                              dt::Float64, time::Float64)
    @inbounds for (index, descriptor) in enumerate(workspace.descriptors)
        if !workspace.online[descriptor.table_index]
            workspace.modes[index] = LIMIT_FREE
            continue
        end
        value = z[descriptor.state_index]
        lower = p[descriptor.lower_parameter_index]
        upper = p[descriptor.upper_parameter_index]
        previous = workspace.modes[index]
        free = _free_limit_residual(descriptor, z, zold, p, dt)
        mode = if previous == LIMIT_UPPER
            value < upper - workspace.tolerance || free > workspace.tolerance ? LIMIT_FREE : LIMIT_UPPER
        elseif previous == LIMIT_LOWER
            value > lower + workspace.tolerance || free < -workspace.tolerance ? LIMIT_FREE : LIMIT_LOWER
        elseif value >= upper - workspace.tolerance && free <= workspace.tolerance
            LIMIT_UPPER
        elseif value <= lower + workspace.tolerance && free >= -workspace.tolerance
            LIMIT_LOWER
        else
            LIMIT_FREE
        end
        mode == previous && continue
        if previous != LIMIT_FREE
            side = previous == LIMIT_LOWER ? :lower : :upper
            bound = previous == LIMIT_LOWER ? lower : upper
            push!(workspace.events, LimitEvent(descriptor.device_type, descriptor.bus,
                descriptor.device_id, descriptor.state_index, side, :release,
                time, bound, value))
        end
        if mode != LIMIT_FREE
            side = mode == LIMIT_LOWER ? :lower : :upper
            bound = mode == LIMIT_LOWER ? lower : upper
            push!(workspace.events, LimitEvent(descriptor.device_type, descriptor.bus,
                descriptor.device_id, descriptor.state_index, side, :activate,
                time, bound, value))
        end
        workspace.modes[index] = mode
    end
    return nothing
end
