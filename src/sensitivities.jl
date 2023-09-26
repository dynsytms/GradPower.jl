# Implementation of the (continous) tangent linear model (TLM) using a backward-Euler integration
# scheme.
function tlm(
    δz0::AbstractArray,
    dp::DynamicProblem,
    ps::PowerSystem,
    traj::AbstractArray,
    tvec::AbstractArray;
    store_trajectory=false,
    δp::Union{AbstractArray, Nothing}=nothing,
    finite_diff::Bool=false
)
    nbus = length(ps.buses)
    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    system_size = diff_dim + alg_dim + 2*nbus

    # integration time is determined by trajectory.
    @assert length(tvec) == size(traj, 2)
    @assert size(traj, 1) == length(δz0)
    @assert size(δz0, 1) == system_size

    # fixed time step (for now)
    dt = tvec[2] - tvec[1]
    tf = tvec[end]
    nsteps = size(tvec, 1)

    # retrieve events.
    # TODO: we can only do one event right now.
    events = ps.dynamic.events
    @assert length(events) <= 1
    ton = toff = tf + dt
    step_on = step_off = nsteps + 1
    if length(events) == 1
        event = events[1]
        ton = event.ton
        toff = event.toff
        step_on = Int(round(ton/dt))
        step_off = Int(round(toff/dt))
    end

    # buffers
    δz = copy(δz0)
    J = preallocate_jacobian(ps)
    rhs = zeros(system_size)
    dt = tvec[2] - tvec[1]
    @views beuler_sens_jac!(J, traj[:, 1], dp.uvec, dp.pvec, ps, diff_dim, dt)
    fact = klu(J)

    for i = 1:(nsteps - 1)
        @views beuler_sens_jac!(J, traj[:, i + 1], dp.uvec, dp.pvec, ps, diff_dim, dt)
        rhs .= 0.0
        if δp != nothing
            if finite_diff
                jacp_vec_fd!(rhs, δp, dp.zvec, dp.uvec, dp.pvec, ps)
            else
                jacp_vec!(rhs, δp, dp.zvec, dp.uvec, dp.pvec, ps)
            end
        end
        @views rhs[1:diff_dim] .*= dt
        @views rhs[1:diff_dim] .+= δz[1:diff_dim]
        rhs .*= -1.0
        δz .= J \ rhs
        #klu!(fact, J)
        #ldiv!(δz, fact, rhs)
        if i == step_on
            activate!(events[1])
        elseif i == step_off
            deactivate!(events[1])
        end
    end

    for event in events
        deactivate!(event)
    end

    return δz
end

function rdiff!(
    out::AbstractArray,
    z::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    ps::PowerSystem
)
    # for now, just quadratic diagonal of speeds.

    out[5] = 2*z[5]
end

function adjoint(
    λ0::AbstractArray,
    dp::DynamicProblem,
    ps::PowerSystem,
    traj::AbstractArray,
    tvec::AbstractArray;
    store_trajectory=false,
    δp::Union{AbstractArray, Nothing}=nothing,
    finite_diff::Bool=false,
    functional::Bool=false
)
    nbus = length(ps.buses)
    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    system_size = diff_dim + alg_dim + 2*nbus

    # integration time is determined by trajectory.
    @assert length(tvec) == size(traj, 2)
    @assert size(traj, 1) == length(λ0)
    @assert size(λ0, 1) == system_size

    # fixed time step (for now)
    dt = tvec[2] - tvec[1]
    tf = tvec[end]
    nsteps = size(tvec, 1)

    # buffers
    λ = copy(λ0)
    μ = zeros(size(dp.pvec, 1))
    J = preallocate_jacobian(ps)
    rhs = zeros(system_size)
    dt = tvec[2] - tvec[1]

    for i = 1:(nsteps - 1)
        rhs .= 0.0

        # construct transpose of Jacobian
        fill!(J.nzval, 0.0)
        rhs_jac!(J, traj[:, end - (i - 1)], dp.uvec, dp.pvec, ps)
        Jt = transpose(J)

        # TODO: INNEFFICIENT: modify Jacobian.
        Jt[1:diff_dim, :] .*= dt
        for j = 1:diff_dim
            Jt[j, j] -= 1.0
        end
        
        # assemble r.h. s
        if functional == true
            rdiff!(rhs, traj[:, end - (i - 1)], dp.uvec, dp.pvec, ps)
        end
        rhs[1:diff_dim] *= dt

        @views rhs[1:diff_dim] .+= -λ[1:diff_dim]
        λ .= Jt \ rhs

        # update μ
        outp = zeros(size(dp.pvec, 1))
        jacpt_vec_fd!(outp, λ, dp.zvec, dp.uvec, dp.pvec, ps)
        μ = μ + dt*outp
    end

    return λ, μ
end

# TODO: Unify beuler methods. This is just negative sign.
function beuler_sens_jac!(
    J::SparseMatrixCSC,
    z::AbstractVector,
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
    _jacobian_sens_beuler!(J, diff_dim, dt)
end

function _jacobian_sens_beuler!(J::SparseMatrixCSC, NDIFFEQ::Int, h::Float64)
    # Iterating through each column
    for col = 1:size(J, 2)
        # Flag to check if diagonal element for the column is found
        diagonal_found = false

        # Iterating through the non-zero elements in each column
        for row_index in nzrange(J, col)
            row = rowvals(J)[row_index]

            # Update values if the row index is less or equal to NDIFFEQ
            if row <= NDIFFEQ
                J.nzval[row_index] *= h

                # Update diagonal element
                if row == col
                    J.nzval[row_index] -= 1.0
                    diagonal_found = true
                end
           end
        end

        # If diagonal element was not found and col is within NDIFFEQ, then add it
        if !diagonal_found && col <= NDIFFEQ
            @warn "Diagonal element not found for column $col. Adding it."
            J[col, col] -= 1.0
        end
    end
end
