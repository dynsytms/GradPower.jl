# Exact control events are located on the incoming smooth branch. Extending
# that branch for a trial solve prevents a semismooth corrector from silently
# jumping over a limit-induced maximum or selecting a different solution.
struct CPFControlEvent
    bus::Int
    kind::Symbol
    lambda::Float64
    state::Vector{Float64}
    incoming::Vector{Float64}
    outgoing::Vector{Float64}
    error::Float64
end

struct CPFControlTrace
    points::Vector{CPFPoint}
    events::Vector{CPFControlEvent}
    status::Symbol
    rejected_steps::Int
    diagnostic::NamedTuple
end

function _cpf_branches(w, x)
    m = length(w.active)
    return [w.lower[k] == w.upper[k] ? :fixed :
            x[2m+k]-w.control_scale*(x[m+w.index[bus]]-w.vset[bus]) <= w.lower[k] ? :lower :
            x[2m+k]-w.control_scale*(x[m+w.index[bus]]-w.vset[bus]) >= w.upper[k] ? :upper : :free
            for (k,bus) in enumerate(w.controlled)]
end

# (control index, destination branch, event kind). Feasible margins are >=0.
function _cpf_guards(w, branches)
    guards = Tuple{Int,Symbol,Symbol}[]
    for (k, branch) in enumerate(branches)
        if branch === :free
            isfinite(w.lower[k]) && push!(guards,(k,:lower,:qmin))
            isfinite(w.upper[k]) && push!(guards,(k,:upper,:qmax))
        elseif branch in (:lower,:upper)
            push!(guards,(k,:free,:recovery))
        end
    end
    return guards
end

function _cpf_margin(w, x, guard, branches)
    k,destination,_ = guard
    m = length(w.active)
    if branches[k] === :free
        return destination === :lower ? x[2m+k]-w.lower[k] : w.upper[k]-x[2m+k]
    end
    bus = w.controlled[k]
    h = w.control_scale*(x[m+w.index[bus]]-w.vset[bus])
    return branches[k] === :lower ? h : -h
end

function _cpf_margin_derivative(w, tangent, guard, branches)
    k,destination,_ = guard
    m = length(w.active)
    if branches[k] === :free
        return destination === :lower ? tangent[2m+k] : -tangent[2m+k]
    end
    dv = w.control_scale*tangent[m+w.index[w.controlled[k]]]
    return branches[k] === :lower ? dv : -dv
end

function _cpf_locate_control(w, previous, step, guard, branches, weights;
                             tol, event_tol, max_newton, lower_step=0.)
    z0 = vcat(previous.state,previous.lambda)
    row = weights.*previous.tangent
    lo,hi = lower_step,step
    # Root bracketing is in the predictor's local arclength coordinate.
    for _ in 1:60
        s = (lo+hi)/2
        solution = _cpf_correct(w,z0+s*previous.tangent,row;tol,max_newton,branches)
        solution === nothing && return nothing
        z,it,residual = solution
        f = _cpf_margin(w,@view(z[1:end-1]),guard,branches)
        # Localize in both margin and section coordinate for reliable ordering.
        abs(f) <= tol && hi-lo <= event_tol && return (s,z,it,residual,abs(f))
        if f >= 0
            lo = s
        else
            hi = s
        end
    end
    return nothing
end

# Determine a consistent outgoing one-sided derivative at a kink. Simultaneous
# events are switched together; further zero-margin guards are checked against
# the resulting tangent. Failure or a cycle is explicit, never a guessed nose.
function _cpf_depart(w, point, branches, guards, weights; event_tol, max_iterations=2length(branches)+2)
    outgoing = copy(branches)
    changed = copy(guards)
    for guard in changed
        outgoing[guard[1]] = guard[2]
    end
    visited = Set{Tuple}()
    failure(status) = (;status,branches=copy(outgoing),visited=collect(visited))
    for _ in 1:max_iterations
        key = Tuple(outgoing)
        key in visited && return failure(:control_cone_cycle)
        push!(visited,key)
        tangent = _cpf_tangent(w,point.state,point.tangent,weights;branches=outgoing)
        if tangent === nothing
            # A singular orientation border is not necessarily a singular
            # branch: the incoming tangent may be orthogonal to the outgoing
            # one. Inspect the rectangular derivative before diagnosing rank.
            A = hcat(Matrix(_cpf_jacobian!(w,point.state;branches=outgoing)),-w.direction)
            D = svd(A;full=true)
            rank = count(s -> s > 1e-10*max(first(D.S),eps()),D.S)
            rank == size(A,1) || return merge(failure(:control_cone_singular),
                (nullity=size(A,2)-rank,))
            tangent = D.V[:,end]
            tangent ./= _cpf_norm(tangent,weights)
            dot(tangent,weights.*point.tangent) < 0 && (tangent .*= -1)
        end
        boundary = [g for g in _cpf_guards(w,outgoing)
                    if abs(_cpf_margin(w,point.state,g,outgoing)) <= event_tol]
        # At a limit-induced maximum, orientation by the incoming dot product
        # alone can select the infeasible half of the outgoing branch. Choose
        # the feasible side of the control cone, even across an obtuse kink.
        if !isempty(boundary) &&
           all(g -> _cpf_margin_derivative(w,tangent,g,outgoing) <= event_tol,boundary) &&
           any(g -> _cpf_margin_derivative(w,tangent,g,outgoing) < -event_tol,boundary)
            tangent .*= -1
        end
        extra = [g for g in boundary if _cpf_margin_derivative(w,tangent,g,outgoing) < -event_tol]
        if isempty(extra)
            any(g -> abs(_cpf_margin_derivative(w,tangent,g,outgoing)) <= event_tol,boundary) &&
                return failure(:control_cone_grazing)
            outgoing == branches && return failure(:control_cone_cycle)
            # Report actual one-sided regime changes, not transient pivots
            # made while solving the cone consistency problem.
            transitions = [(k,outgoing[k],outgoing[k] === :free ? :recovery :
                            outgoing[k] === :upper ? :qmax : :qmin)
                           for k in eachindex(branches) if branches[k] != outgoing[k]]
            return (status=:ok,branches=outgoing,tangent=tangent,changed=transitions)
        end
        for guard in extra
            outgoing[guard[1]] = guard[2]
            push!(changed,guard)
        end
    end
    return failure(:control_cone_iteration_limit)
end

function _cpf_trace_controls(ps, direction=LoadingDirection(ps);
                              ds0=0.05, ds_min=1e-5, ds_max=0.2, tol=1e-9,
                              event_tol=1e-7, predictor_tol=1e-3,
                              max_steps=500, max_newton=12, max_retries=12,
                              stop=:nose, control_scale=1.0,
                              observer=nothing, retain_history=true, stats=nothing)
    all(v -> isfinite(v) && v>0, (ds0,ds_min,ds_max,tol,event_tol,predictor_tol)) ||
        throw(ArgumentError("step sizes and tolerances must be finite and positive"))
    ds_min <= ds0 <= ds_max || throw(ArgumentError("require ds_min <= ds0 <= ds_max"))
    event_tol >= tol || throw(ArgumentError("event_tol must be at least tol"))
    max_steps >= 0 && max_newton > 0 && max_retries >= 0 || throw(ArgumentError("invalid iteration budget"))
    stop in (:nose,:steps) || throw(ArgumentError("stop must be :nose or :steps"))
    w = _cpf_workspace(ps,direction;control_scale,stats)
    any(!iszero,w.direction) || throw(ArgumentError("continuation direction is zero"))
    points,events = CPFPoint[],CPFControlEvent[]
    result(status,rejected,diagnostic=(;)) = CPFControlTrace(points,events,status,rejected,diagnostic)
    function finish_fold(fold,rejected,branches)
        certificate = _cpf_verify_fold(w,fold;branches)
        _cpf_accept!(points,fold,w,observer,retain_history;branches)
        status = certificate.status === :simple_fold ? :smooth_fold : certificate.status
        return result(status,rejected,certificate)
    end
    weights = _cpf_weights(w)
    reference = zeros(length(w.initial)+1)
    reference[end] = 1.
    initial = _cpf_correct(w,vcat(w.initial,0.),reference;tol,max_newton)
    initial === nothing && return result(:initial_failure,0)
    z,it,residual = initial
    branches = _cpf_branches(w,@view z[1:end-1])
    point = _cpf_point(w,z,reference,weights,it,residual;branches)
    point === nothing && return result(:tangent_failure,0)
    _cpf_accept!(points,point,w,observer,retain_history;branches)
    ds,rejected = ds0,0
    for _ in 1:max_steps
        previous = last(points)
        guards = _cpf_guards(w,branches)
        # A base solution or a previous event may lie exactly on another
        # boundary. Resolve its outgoing side before predicting a new point.
        immediate = [g for g in guards if abs(_cpf_margin(w,previous.state,g,branches)) <= event_tol &&
                     _cpf_margin_derivative(w,previous.tangent,g,branches) < -event_tol]
        if !isempty(immediate)
            departure = _cpf_depart(w,previous,branches,immediate,weights;event_tol)
            departure.status === :ok || return result(departure.status,rejected,departure)
            outgoing,tangent,changed = departure.branches,departure.tangent,departure.changed
            if stop === :nose && min(abs(previous.tangent[end]),abs(tangent[end])) <= event_tol
                return result(:fold_control_coincidence,rejected,
                    (event=previous,incoming=copy(previous.tangent),outgoing=copy(tangent)))
            end
            for guard in changed
                k,_,kind = guard
                event = CPFControlEvent(w.buses[w.controlled[k]],kind,previous.lambda,
                    copy(previous.state),copy(previous.tangent),copy(tangent),
                    abs(_cpf_margin(w,previous.state,guard,branches)))
                observer === nothing || observer(:event,w,event,outgoing)
                retain_history && push!(events,event)
            end
            points[end] = CPFPoint(previous.state,previous.lambda,tangent,previous.residual,previous.iterations)
            observer === nothing || observer(:replace,w,points[end],outgoing)
            if previous.tangent[end] > event_tol && tangent[end] < -event_tol && stop === :nose
                return result(:limit_induced_maximum,rejected)
            end
            branches = outgoing
            previous = last(points)
            guards = _cpf_guards(w,branches)
        end
        z0 = vcat(previous.state,previous.lambda)
        candidate = nothing
        predictor_error = 0.
        for retry in 0:max_retries
            predicted = z0+ds*previous.tangent
            solution = _cpf_correct(w,predicted,weights.*previous.tangent;tol,max_newton,branches)
            if solution !== nothing
                z,it,residual = solution
                candidate = _cpf_point(w,z,previous.tangent,weights,it,residual;branches)
                predictor_error = _cpf_norm(z-predicted,weights)
                if candidate !== nothing && predictor_error <= 2predictor_tol &&
                   dot(candidate.tangent,weights.*previous.tangent) >= .5
                    break
                end
            end
            candidate = nothing
            rejected += 1
            retry == max_retries && return result(:retry_limit,rejected)
            ds /= 2
            ds < ds_min && return result(:minimum_step,rejected)
        end
        scan = isempty(guards) ? (status=:ok,hits=[]) :
            _cpf_scan_controls(w,previous,candidate,ds,guards,branches,weights;tol,event_tol,max_newton)
        scan.status === :ok || return result(scan.status,rejected,scan)
        crossed = scan.hits
        fold = nothing
        if previous.tangent[end] > 0 && candidate.tangent[end] <= 0 && stop === :nose
            fold = _cpf_refine_fold(w,previous,candidate,weights;tol,event_tol,max_newton,branches)
            fold === nothing && return result(:fold_refinement_failure,rejected)
        end
        if !isempty(crossed)
            located = [_cpf_locate_control(w,previous,hit.hi,hit.guard,branches,weights;
                        tol,event_tol,max_newton,lower_step=hit.lo) for hit in crossed]
            any(isnothing,located) && return result(:event_localization_failure,rejected)
            first_event = argmin(first.(located))
            s,z,it,residual,_ = located[first_event]
            if fold !== nothing
                fold_step = dot(weights.*previous.tangent,vcat(fold.state,fold.lambda)-z0)
                separation = fold_step-s
                if abs(separation) <= 4event_tol
                    return result(:fold_control_coincidence,rejected,
                        (fold=fold,event_state=copy(z),separation=separation))
                elseif separation < 0
                    margins = [_cpf_margin(w,fold.state,g,branches) for g in guards]
                    minimum(margins) > event_tol && return finish_fold(fold,rejected,branches)
                    # Even when coordinates separate, a near-zero guard at
                    # the fold cannot certify that it is a smooth interior event.
                    return result(:fold_control_coincidence,rejected,
                        (fold=fold,event_state=copy(z),separation=separation,margins=margins))
                end
            end
            incoming = _cpf_point(w,z,previous.tangent,weights,it,residual;branches)
            incoming === nothing && return result(:tangent_failure,rejected)
            # Include other boundaries at this same corrected point, not just
            # those violated by the endpoint (simultaneous events).
            simultaneous = [g for g in guards if abs(_cpf_margin(w,incoming.state,g,branches)) <= event_tol &&
                            _cpf_margin_derivative(w,incoming.tangent,g,branches) < -event_tol]
            isempty(simultaneous) && return result(:event_localization_failure,rejected)
            departure = _cpf_depart(w,incoming,branches,simultaneous,weights;event_tol)
            departure.status === :ok || return result(departure.status,rejected,departure)
            outgoing,tangent,changed = departure.branches,departure.tangent,departure.changed
            if stop === :nose && min(abs(incoming.tangent[end]),abs(tangent[end])) <= event_tol
                return result(:fold_control_coincidence,rejected,
                    (event=incoming,incoming=copy(incoming.tangent),outgoing=copy(tangent)))
            end
            exact = _cpf_residual!(zeros(length(incoming.state)),w,incoming.state,incoming.lambda)
            norm(exact,Inf) <= 2event_tol || return result(:event_feasibility_failure,rejected)
            for guard in changed
                k,_,kind = guard
                event = CPFControlEvent(w.buses[w.controlled[k]],kind,incoming.lambda,
                    copy(incoming.state),copy(incoming.tangent),copy(tangent),
                    abs(_cpf_margin(w,incoming.state,guard,branches)))
                observer === nothing || observer(:event,w,event,outgoing)
                retain_history && push!(events,event)
            end
            _cpf_accept!(points,CPFPoint(incoming.state,incoming.lambda,tangent,norm(exact,Inf),it),
                         w,observer,retain_history;branches=outgoing)
            if incoming.tangent[end] > event_tol && tangent[end] < -event_tol && stop === :nose
                return result(:limit_induced_maximum,rejected)
            end
            branches = outgoing
            ds = clamp(min(ds,max(s,ds_min)),ds_min,ds_max)
            continue
        end
        if fold !== nothing
            # The fold must lie on the feasible portion of this regime.
            any(g -> _cpf_margin(w,fold.state,g,branches)<-event_tol,guards) &&
                return result(:event_localization_failure,rejected)
            near = [g for g in guards if abs(_cpf_margin(w,fold.state,g,branches)) <= event_tol]
            isempty(near) || return result(:fold_control_coincidence,rejected,(fold=fold,guards=near))
            return finish_fold(fold,rejected,branches)
        end
        _cpf_accept!(points,candidate,w,observer,retain_history;branches)
        factor = clamp(sqrt(predictor_tol/max(predictor_error,eps())),.5,1.5)
        ds = clamp(ds*factor,ds_min,ds_max)
    end
    return result(:max_steps,rejected)
end
