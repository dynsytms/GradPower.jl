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
)
    nbus = length(ps.buses)
    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    system_size = diff_dim + alg_dim + 2*nbus

    # integration time is determined by trajectory.
    @assert length(tvec) == size(traj, 2)
    @assert size(traj, 1) == length(δz0)
    @assert size(δz0, 1) == system_size

    # buffers
    δz = copy(δz0)
    J = preallocate_jacobian(ps)
    rhs = zeros(system_size)

    nsteps = size(tvec, 1)
    for i = 1:(nsteps - 1)
        dt = tvec[i+1] - tvec[i]
        @views beuler_jac!(J, traj[:, i + 1], traj[:, i + 1], dp.uvec, dp.pvec, ps, diff_dim, dt)
            
        rhs .= 0.0
        if δp != nothing
            jacp_vec!(rhs, δp, dprob.zvec, dprob.uvec, dprob.pvec, ps)
        end

        @views rhs[1:diff_dim] .*= dt
        @views rhs[1:diff_dim] .+= δz[1:diff_dim]
        rhs .*= -1.0
        δz = J \ rhs
        #fact = klu(J)
        #klu!(fact, J)
        #ldiv!(δz, fact, rhs)
    end

    return δz
end
