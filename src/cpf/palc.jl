# Smooth CPU continuation engine; public recording is supplied by an observer.
struct CPFPoint
    state::Vector{Float64}
    lambda::Float64
    tangent::Vector{Float64}
    residual::Float64
    iterations::Int
end

struct CPFTrace
    points::Vector{CPFPoint}
    status::Symbol
    rejected_steps::Int
    diagnostic::NamedTuple
end
CPFTrace(points,status,rejected_steps) = CPFTrace(points,status,rejected_steps,(;))

function _cpf_accept!(points, point, w, observer, retain_history; branches=nothing)
    observer === nothing || observer(:point,w,point,branches)
    retain_history || empty!(points)
    push!(points,point)
    return point
end

_cpf_norm(x, weights) = sqrt(dot(x, weights .* x))

function _cpf_weights(w::CPFWorkspace)
    m = length(w.active)
    nc = length(w.controlled)
    # Each physical block has a size-normalized contribution. Lambda has
    # unit weight; Q shares the same nominal pu scale as voltage and angle.
    return vcat(fill(1.0/m, 2m), fill(1.0/max(nc,1), nc), 1.0)
end

function _cpf_augmented(w, x, row; branches=nothing)
    J = _cpf_jacobian!(w, x; branches)
    return [J sparse(reshape(-w.direction, :, 1)); sparse(reshape(row, 1, :))]
end

function _cpf_linear(A, rhs, stats=nothing)
    # Full augmentation remains regular at a simple fold even when F_x is
    # singular. Do not eliminate the parameter using inverse(F_x).
    try
        started = stats === nothing ? UInt64(0) : time_ns()
        stats === nothing || (stats.factorizations += 1)
        factor = try
            klu(A)
        finally
            stats === nothing || (stats.factorization_ns += time_ns()-started)
        end
        if stats !== nothing
            stats.solves += 1
            started = time_ns()
        end
        delta = try
            factor \ rhs
        finally
            stats === nothing || (stats.solve_ns += time_ns()-started)
        end
        all(isfinite, delta) || return nothing
        norm(A*delta-rhs, Inf) <= 1e-7 * max(1., norm(rhs,Inf)) || return nothing
        return delta
    catch err
        err isa LinearAlgebra.SingularException || err isa LinearAlgebra.ZeroPivotException || rethrow()
        return nothing
    end
end

function _cpf_tangent(w, x, previous, weights; branches=nothing)
    rhs = zeros(length(x)+1)
    rhs[end] = 1.
    t = _cpf_linear(_cpf_augmented(w,x,weights.*previous; branches),rhs,w.stats)
    t === nothing && return nothing
    t ./= _cpf_norm(t,weights)
    dot(t,weights.*previous) < 0 && (t .*= -1)
    return t
end

# Correct on a fixed hyperplane. Initial solve uses the plane lambda=0.
# Rejected steps use fresh arrays, never altering the last accepted point.
function _cpf_correct(w, predicted, row; tol, max_newton, branches=nothing)
    z = copy(predicted)
    n = length(z)-1
    function residual!(out, z)
        _cpf_residual!(@view(out[1:n]), w, @view(z[1:n]), z[end]; branches)
        out[end] = dot(row,z-predicted)
        return norm(out,Inf)
    end
    r, trial_r = zeros(n+1), zeros(n+1)
    for it in 0:max_newton
        err = residual!(r,z)
        isfinite(err) || return nothing
        if err <= tol
            return (z,it,maximum(abs, @view(r[1:n])))
        end
        it == max_newton && return nothing
        delta = _cpf_linear(_cpf_augmented(w,@view(z[1:n]),row; branches),-r,w.stats)
        delta === nothing && return nothing
        alpha = 1.
        accepted = false
        for _ in 1:12
            trial = z + alpha*delta
            m = length(w.active)
            if minimum(@view trial[m+1:2m]) > 0
                trial_err = residual!(trial_r,trial)
                if isfinite(trial_err) && trial_err < (1-1e-4*alpha)*err
                    z = trial
                    accepted = true
                    break
                end
            end
            alpha *= .5
        end
        accepted || return nothing
    end
    return nothing
end

function _cpf_point(w,z,previous,weights,it,residual; branches=nothing)
    t = _cpf_tangent(w,@view(z[1:end-1]),previous,weights; branches)
    t === nothing && return nothing
    return CPFPoint(copy(z[1:end-1]),z[end],t,residual,it)
end

# Root-find t_lambda in a bracket of corrected points. A fixed local section
# parameterizes the bracket even when lambda is not monotone. Every root
# evaluation is corrected; interpolated samples are never called solutions.
function _cpf_refine_fold(w,left,right,weights;tol,event_tol,max_newton,branches=nothing)
    a = vcat(left.state,left.lambda)
    b = vcat(right.state,right.lambda)
    secant = b-a
    row = weights .* (secant/_cpf_norm(secant,weights))
    low, high = 0., 1.
    fl, fh = left.tangent[end], right.tangent[end]
    span = _cpf_norm(secant,weights)
    for _ in 1:60
        u = clamp((low*fh-high*fl)/(fh-fl),low+.1*(high-low),high-.1*(high-low))
        corrected = _cpf_correct(w,a+u*secant,row;tol,max_newton,branches)
        corrected === nothing && return nothing
        z,it,residual = corrected
        point = _cpf_point(w,z,left.tangent,weights,it,residual;branches)
        point === nothing && return nothing
        # Bound the location as well as the tangent residual. A small
        # t_lambda alone gives no coordinate accuracy near a flat maximum.
        abs(point.tangent[end]) <= event_tol && (high-low)*span <= event_tol && return point
        if point.tangent[end] > 0
            low,fl = u,point.tangent[end]
        else
            high,fh = u,point.tangent[end]
        end
    end
    return nothing
end

function _cpf_trace_smooth(ps, direction=LoadingDirection(ps);
                           ds0=0.05, ds_min=1e-5, ds_max=0.2, tol=1e-9,
                           event_tol=1e-7, predictor_tol=1e-3,
                           max_steps=500, max_newton=12, max_retries=12,
                           stop=:nose, observer=nothing, retain_history=true, stats=nothing)
    all(v -> isfinite(v) && v>0, (ds0,ds_min,ds_max,tol,event_tol,predictor_tol)) ||
        throw(ArgumentError("step sizes and tolerances must be finite and positive"))
    ds_min <= ds0 <= ds_max || throw(ArgumentError("require ds_min <= ds0 <= ds_max"))
    max_steps >= 0 && max_newton > 0 && max_retries >= 0 || throw(ArgumentError("invalid iteration budget"))
    stop in (:nose,:steps) || throw(ArgumentError("stop must be :nose or :steps"))
    w = _cpf_workspace(ps,direction;limits=:none,stats)
    any(!iszero,w.direction) || throw(ArgumentError("continuation direction is zero"))
    weights = _cpf_weights(w)
    reference = zeros(length(w.initial)+1)
    reference[end] = 1.
    initial = _cpf_correct(w,vcat(w.initial,0.),reference;tol,max_newton)
    initial === nothing && return CPFTrace(CPFPoint[],:initial_failure,0)
    z,it,residual = initial
    point = _cpf_point(w,z,reference,weights,it,residual)
    point === nothing && return CPFTrace(CPFPoint[],:tangent_failure,0)
    points = [point]
    observer === nothing || observer(:point,w,point,nothing)
    rejected = 0
    ds = ds0
    for _ in 1:max_steps
        previous = last(points)
        z0 = vcat(previous.state,previous.lambda)
        next = nothing
        predictor_error = 0.
        for retry in 0:max_retries
            predicted = z0 + ds*previous.tangent
            solution = _cpf_correct(w,predicted,weights.*previous.tangent;tol,max_newton)
            if solution !== nothing
                z,it,residual = solution
                candidate = _cpf_point(w,z,previous.tangent,weights,it,residual)
                predictor_error = _cpf_norm(z-predicted,weights)
                if candidate !== nothing && predictor_error <= 2predictor_tol &&
                   dot(candidate.tangent,weights.*previous.tangent) >= .5
                    next = candidate
                    break
                end
            end
            rejected += 1
            retry == max_retries && return CPFTrace(points,:retry_limit,rejected)
            ds /= 2
            ds < ds_min && return CPFTrace(points,:minimum_step,rejected)
        end
        if previous.tangent[end] > 0 && next.tangent[end] <= 0 && stop === :nose
            fold = _cpf_refine_fold(w,previous,next,weights;tol,event_tol,max_newton)
            fold === nothing && return CPFTrace(points,:fold_refinement_failure,rejected)
            certificate = _cpf_verify_fold(w,fold)
            _cpf_accept!(points,fold,w,observer,retain_history)
            status = certificate.status === :simple_fold ? :smooth_fold : certificate.status
            return CPFTrace(points,status,rejected,certificate)
        end
        _cpf_accept!(points,next,w,observer,retain_history)
        factor = clamp(sqrt(predictor_tol/max(predictor_error,eps())),.5,1.5)
        ds = clamp(ds*factor,ds_min,ds_max)
    end
    return CPFTrace(points,:max_steps,rejected)
end
