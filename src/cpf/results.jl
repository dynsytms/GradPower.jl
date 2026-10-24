"""
    ContinuationPoint

One accepted operating point. `vm`, `va`, `pinj`, `qinj`, `pg`, `qg`, and `control` follow
`result.bus_numbers`. Magnitudes are pu, angles radians, and generation is
aggregate **online generation**, pu on the system base. Reference dispatch
includes the balancing mismatch. Scheduled generation excludes the exogenous
net-injection direction; only reference P/Q and regulating PV Q are solved.
`pinj`/`qinj` are total network injections, including the direction, not demands.
`control` is `:pq`, `:reference`, `:voltage`, `:qmin`, `:qmax`, or `:fixed_q`.
`residual` is the infinity norm of the full solver residual (not just recorded
buses); `iterations` counts corrector Newton updates.
"""
struct ContinuationPoint
    lambda::Float64
    vm::Vector{Float64}
    va::Vector{Float64}
    pinj::Vector{Float64}
    qinj::Vector{Float64}
    pg::Vector{Float64}
    qg::Vector{Float64}
    control::Vector{Symbol}
    residual::Float64
    iterations::Int
end

"""
    ContinuationEvent

Located reactive control exchange. `bus` is an external bus number, even if
that bus was not selected for recording. `kind` is `:qmin`, `:qmax`, or
`:recovery`. `point` records selected physical channels at the event.
`incoming` and `outgoing` are one-sided tangents with fields `va`, `vm`,
`qg` (selected aggregate controlled-Q derivatives), and `lambda`. Derivatives
are with respect to the solver's weighted arclength, not the loading factor.
Reference-Q tangent entries are `NaN` (not computed). `error` is the absolute
control guard residual, not a bound on loading-factor error.
"""
struct ContinuationEvent
    bus::Int
    kind::Symbol
    point::ContinuationPoint
    incoming::NamedTuple
    outgoing::NamedTuple
    error::Float64
end

"""
    ContinuationResult

CPU continuation trace returned by [`runcpf`](@ref).

- `bus_numbers`: external bus numbers, in recording order.
- `points`, `events`: accepted physical points and located control exchanges.
- `status`, `message`, `diagnostic`: termination reason and numerical details.
  Diagnostic keys are status-dependent, implementation-level details; only
  the termination status and physical records are stable public interfaces.
  Degeneracy diagnostics can contain full internal states even with selected
  recording (bounded terminal context, not an all-bus history).
- `maximum`: refined terminal point only for `:smooth_fold` (numerically
  verified simple fold) or `:limit_induced_maximum` (one-sided loading reversal).
  Otherwise `nothing`, including after a fixed step budget.
- `sampled_maximum`: highest-loading accepted recorded point, or `nothing`.
  This is **not** a verified loading margin or a global optimum.
- `success`: true for a verified local maximum, or a completed `stop=:steps`
  budget. `:max_steps` with `stop=:nose` is an incomplete trace, not success.
- `rejected_steps`, `baseMVA`, `limits`, `stop`: run metadata.

Numerical failures preserve partial accepted results instead of throwing.
Invalid input throws `ArgumentError`. Numerical fold checks are tolerance-based,
not rigorous certificates or claims of global branch uniqueness.
"""
struct ContinuationResult
    bus_numbers::Vector{Int}
    points::Vector{ContinuationPoint}
    events::Vector{ContinuationEvent}
    status::Symbol
    message::String
    diagnostic::NamedTuple
    maximum::Union{Nothing,ContinuationPoint}
    sampled_maximum::Union{Nothing,ContinuationPoint}
    success::Bool
    rejected_steps::Int
    baseMVA::Float64
    limits::Symbol
    stop::Symbol
end

function _cpf_recording_indices(ps, buses)
    ids = [b.i for b in ps.buses]
    length(unique(ids)) == length(ids) || throw(ArgumentError("external bus numbers must be unique"))
    buses isa Colon && return collect(eachindex(ps.buses))
    buses isa AbstractVector || throw(ArgumentError("buses must be : or a vector of external bus numbers"))
    selected = Int[]
    for bus in buses
        bus isa Integer && !(bus isa Bool) || throw(ArgumentError("bus numbers must be integers"))
        i = findfirst(==(bus),ids)
        i === nothing && throw(ArgumentError("unknown external bus number $bus"))
        i in selected && throw(ArgumentError("duplicate recording bus $bus"))
        push!(selected,i)
    end
    return selected
end

# Snapshot demands and dispatch independently of internal residual conventions.
function _cpf_physical_point(ps, w, x, lambda, indices, residual, iterations; branches=nothing)
    vm,va = _cpf_voltages(w,x)
    v = vm .* cis.(va)
    net = v .* conj.(w.ybus*v)
    pd,qd,pg,qg = zeros(length(vm)),zeros(length(vm)),zeros(length(vm)),zeros(length(vm))
    for load in ps.loads
        pd[load.bus] += load.pd
        qd[load.bus] -= load.qd
    end
    for gen in ps.gens
        gen.status || continue
        pg[gen.bus] += gen.psch
        qg[gen.bus] += gen.qsch
    end
    control = fill(:pq,length(vm))
    for (i,bus) in enumerate(ps.buses)
        if bus.type == 3
            pg[i],qg[i] = real(net[i])+pd[i],imag(net[i])+qd[i]
            control[i] = :reference
        end
    end
    m = length(w.active)
    regimes = branches === nothing ? _cpf_branches(w,x) : branches
    for (k,bus) in enumerate(w.controlled)
        # At a PV bus the Q direction is a specified nonregulating injection;
        # the solved Q unknown is the regulating generator aggregate.
        qg[bus] = x[2m+k]
        branch = w.limits === :none ? :free : regimes[k]
        control[bus] = branch === :free ? :voltage : branch === :upper ? :qmax :
                       branch === :lower ? :qmin : :fixed_q
    end
    return ContinuationPoint(lambda,vm[indices],va[indices],real.(net[indices]),imag.(net[indices]),pg[indices],qg[indices],
                             control[indices],residual,iterations)
end

function _cpf_physical_tangent(ps,w,t,indices)
    n,m = length(ps.buses),length(w.active)
    va,vm,qg = zeros(n),zeros(n),zeros(n)
    va[w.active] .= t[1:m]
    vm[w.active] .= t[m+1:2m]
    for (k,bus) in enumerate(w.controlled)
        qg[bus] = t[2m+k]
    end
    qg[findfirst(b -> b.type==3,ps.buses)] = NaN
    return (;va=va[indices],vm=vm[indices],qg=qg[indices],lambda=t[end])
end

function _cpf_status_message(status, stop)
    status === :smooth_fold && return "Located a numerically verified simple smooth loading maximum."
    status === :limit_induced_maximum && return "Located a loading reversal across a reactive-control event."
    status === :max_steps && return stop === :steps ? "Completed the requested continuation step budget." :
        "Step budget exhausted before locating a loading maximum; trace is incomplete."
    status === :initial_failure && return "Initial operating point did not converge; no points accepted."
    status === :minimum_step && return "Required continuation step fell below ds_min; partial trace retained."
    status === :retry_limit && return "Corrector retry budget exhausted; partial trace retained."
    return "Continuation stopped with diagnostic $(status); no verified maximum is reported."
end

"""
    runcpf(ps, direction=LoadingDirection(ps); limits=:complementarity,
           buses=:, stop=:nose, ds0=0.05, ds_min=1e-5, ds_max=0.2,
           tol=1e-9, event_tol=1e-7, predictor_tol=1e-3,
           max_steps=500, max_newton=12, max_retries=12, control_scale=1.0,
           stats=nothing)

Trace steady-state AC solutions without modifying `ps`, its network, dispatch,
loads, or bus types. Returns [`ContinuationResult`](@ref). CPU only; one
connected network with one fixed-voltage reference is supported.

`direction` specifies additive net P/Q injection per internal bus index; see
[`LoadingDirection`](@ref). At a PV bus, a Q direction is a *nonregulating*
specified injection, separate from the bounded regulating generation. At a PQ
bus, the direction is likewise exogenous: scheduled generator dispatch is
unchanged. Total injection is reported separately as `pinj`/`qinj`; the solver
does not infer a load/generation split for a general net direction.
Reference directions must be zero. Reactive limits aggregate online machines;
individual machine allocation is not returned. Reference Q is unrestricted.

`limits=:complementarity` enforces exact Q bounds and two-way control recovery;
`:none` holds PV voltage without enforcing Q bounds. `control_scale` is positive
pu Q per pu voltage; it scales complementarity, not physical limits.

`buses` selects **external** bus numbers in caller order, or `:` for all buses.
An empty vector records scalar diagnostics only. Selection reduces retained
point/event histories during tracing; it does not reduce the full network
solve. All event identities are retained even for unrecorded buses.

`stop=:nose` seeks a local loading maximum; `stop=:steps` traces for at most
`max_steps` accepted advances, including control events, and may follow a lower
branch. A step budget alone does not verify a margin. Initial/base points do
not consume the advance budget. Degenerate events can stop either mode early.

Step sizes use the size-normalized weighted norm of angles, magnitudes,
controlled Q and lambda. `predictor_tol` bounds weighted predictor error;
`tol` bounds the corrector infinity norm. `event_tol >= tol` controls event
location and degeneracy checks. Numerical failures return partial results;
inspect `status` rather than assuming a completed trace. Pass a
[`ContinuationStats`](@ref) as `stats` for cumulative CPU work/timing counters;
logging is disabled by default and does not change solver tolerances.
"""
function runcpf(ps::PowerSystem, direction::LoadingDirection=LoadingDirection(ps);
               limits::Symbol=:complementarity, buses=Colon(), stop=:nose,
               ds0=0.05, ds_min=1e-5, ds_max=0.2, tol=1e-9, event_tol=1e-7,
               predictor_tol=1e-3, max_steps=500, max_newton=12, max_retries=12,
               control_scale=1.0, stats::Union{Nothing,ContinuationStats}=nothing)
    limits in (:none,:complementarity) || throw(ArgumentError("limits must be :none or :complementarity"))
    event_tol >= tol || throw(ArgumentError("event_tol must be at least tol"))
    isfinite(control_scale) && control_scale>0 || throw(ArgumentError("control_scale must be finite and positive"))
    all(x -> x isa Integer && !(x isa Bool), (max_steps,max_newton,max_retries)) ||
        throw(ArgumentError("iteration budgets must be integers"))
    indices = _cpf_recording_indices(ps,buses)
    points,events = ContinuationPoint[],ContinuationEvent[]
    function observe(kind,w,value,branches)
        if kind in (:point,:replace)
            point = _cpf_physical_point(ps,w,value.state,value.lambda,indices,value.residual,value.iterations;branches)
            if kind === :replace
                points[end] = point
            else
                push!(points,point)
            end
        else
            residual = norm(_cpf_residual!(zeros(length(value.state)),w,value.state,value.lambda),Inf)
            point = _cpf_physical_point(ps,w,value.state,value.lambda,indices,residual,0;branches)
            push!(events,ContinuationEvent(value.bus,value.kind,point,
                _cpf_physical_tangent(ps,w,value.incoming,indices),
                _cpf_physical_tangent(ps,w,value.outgoing,indices),value.error))
        end
    end
    options = (;ds0,ds_min,ds_max,tol,event_tol,predictor_tol,max_steps,max_newton,max_retries,
                stop,observer=observe,retain_history=false,stats)
    trace = limits === :none ? _cpf_trace_smooth(ps,direction;options...) :
        _cpf_trace_controls(ps,direction;control_scale,options...)
    verified = trace.status in (:smooth_fold,:limit_induced_maximum)
    maximum = verified ? last(points) : nothing
    sampled = isempty(points) ? nothing : points[argmax(p.lambda for p in points)]
    success = verified || (stop === :steps && trace.status === :max_steps)
    return ContinuationResult([ps.buses[i].i for i in indices],points,events,trace.status,
        _cpf_status_message(trace.status,stop),deepcopy(trace.diagnostic),maximum,sampled,
        success,trace.rejected_steps,ps.baseMVA,limits,stop)
end
