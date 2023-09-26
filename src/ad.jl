function jvp(func, primal, tangent)
    g(t) = func(primal .+ t .* tangent)
    jvp_result = ForwardDiff.derivative(g, 0.0)
    return jvp_result
end


function jacp_vec!(
    out::AbstractArray,
    vec::AbstractArray,
    z::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    sys::PowerSystem;
    full_jac::Bool = false,
)
    map = sys.dynamic.map

    diff_dim = sys.dynamic.diff_dim
    alg_dim = sys.dynamic.alg_dim
    ctrl_dim = sys.dynamic.ctrl_dim
    par_dim = sys.dynamic.par_dim
    dev = diff_dim + alg_dim

    @assert length(vec) == par_dim

    x = @view z[1:diff_dim]
    y = @view z[diff_dim+1:diff_dim+alg_dim]
    v = @view z[diff_dim+alg_dim+1:end]

    out_diff = @view out[1:diff_dim]
    out_alg = @view out[diff_dim+1:diff_dim+alg_dim]


    @inbounds for (i, device) in enumerate(sys.dynamic.devices)
        bus = map.bus[i]

        # retrieve pointers and sizes from map
        diff_ptr = map.diff_ptr[i]
        alg_ptr = map.alg_ptr[i]
        ctrl_ptr = map.ctrl_ptr[i]
        par_ptr = map.par_ptr[i]

        diff_size = map.diff_size[i]
        alg_size = map.alg_size[i]
        ctrl_size = map.ctrl_size[i]
        par_size = map.par_size[i]

        # retrieve local views
        diff = @view x[diff_ptr:diff_ptr+diff_size-1]
        alg = @view y[alg_ptr:alg_ptr+alg_size-1]
        ctrl = @view u[ctrl_ptr:ctrl_ptr+ctrl_size-1]
        par = @view p[par_ptr:par_ptr+par_size-1]
        vloc = @view v[2*bus-1:2*bus]

        # ensure device contributes to derivative. If not, skip
        # We do this by checking that the direction vector is not zero.
        # given that Jvec = 0 if vec = 0.
        if norm(vec[par_ptr:par_ptr+par_size-1]) < 1e-10
            continue
        end

        # Current injection
        function cinj(x)
            f = similar(x, 2)
            fill!(f, zero(eltype(x)))
            GradPower.cinject!(f, diff, alg, ctrl, x, vloc, device.dtype)
            return f
        end

        tangent = @view vec[par_ptr:par_ptr+par_size-1]
        result = @views out[dev + 2*bus - 1: dev + 2*bus]

        if full_jac
            J = ForwardDiff.jacobian(cinj, par)
            result .= J*tangent
        else
            result .= jvp(cinj, par, tangent)
        end

        # Algebraic contribution
        function rhs_g(x)
            f_alg = similar(x, alg_size)
            fill!(f_alg, zero(eltype(x)))
            rhs_alg!(f_alg, diff, alg, ctrl, x, vloc, device.dtype)
            return f_alg
        end

        if alg_size > 0
            tangent = @view vec[par_ptr:par_ptr+par_size-1]
            result = @views out_alg[alg_ptr:alg_ptr+alg_size-1]

            if full_jac
                J = ForwardDiff.jacobian(rhs_g, par)
                result .= J*tangent
            else
                result .= jvp(rhs_g, par, tangent)
            end
        end

        # Differential contribution

        function rhs_f(x)
            f_diff = similar(x, diff_size)
            fill!(f_diff, zero(eltype(x)))
            rhs_diff!(f_diff, diff, alg, ctrl, x, vloc, device.dtype)
            return f_diff
        end

        if diff_size > 0

            tangent = @view vec[par_ptr:par_ptr+par_size-1]
            result = @views out_diff[diff_ptr:diff_ptr+diff_size-1]

            if full_jac
                J = ForwardDiff.jacobian(rhs_f, par)
                result .= J*tangent
            else
                result .= jvp(rhs_f, par, tangent)
            end
        end
    end
end

function jacp_vec_fd!(
    out::AbstractArray,
    vec::AbstractArray,
    z::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    sys::PowerSystem;
)
    function rhs(pnom)
        f = similar(pnom, length(z))
        fill!(f, zero(eltype(pnom)))
        rhs_fun!(f, z, u, pnom, sys)
        return f
    end

    Jfd = FiniteDiff.finite_difference_jacobian(rhs, p)
    out .= Jfd*vec
end

function jacpt_vec_fd!(
    out::AbstractArray,
    vec::AbstractArray,
    z::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    sys::PowerSystem;
)
    function rhs(pnom)
        f = similar(pnom, length(z))
        fill!(f, zero(eltype(pnom)))
        rhs_fun!(f, z, u, pnom, sys)
        return f
    end

    Jfd = FiniteDiff.finite_difference_jacobian(rhs, p)
    out .= transpose(Jfd)*vec
end
