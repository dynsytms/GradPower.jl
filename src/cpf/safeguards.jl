# Exact second directional derivative of the smooth power balances. Control
# equations on a fixed regime and the additive loading direction are affine.
function _cpf_second_direction(w, x, d)
    m = length(w.active)
    mag, angle = _cpf_voltages(w,x)
    da, dv = zeros(length(mag)), zeros(length(mag))
    da[w.active] .= @view d[1:m]
    dv[w.active] .= @view d[m+1:2m]
    phase = cis.(angle)
    v = mag .* phase
    v1 = phase .* (dv + im .* mag .* da)
    v2 = phase .* (2im .* dv .* da - mag .* da.^2)
    s2 = v2 .* conj.(w.ybus*v) + 2v1 .* conj.(w.ybus*v1) + v .* conj.(w.ybus*v2)
    return vcat(real.(s2[w.active]),imag.(s2[w.active]),zeros(length(w.controlled)))
end

# Numerical simple-fold certificate. Dense SVD is intentionally restricted to
# event verification, not continuation steps. Relative tests use unit Euclidean
# null vectors; small coefficients are unresolved, not asserted nonzero.
function _cpf_fold_certificate(J, flambda, second; rank_tol=1e-7, nondeg_tol=1e-6)
    D = svd(Matrix(J))
    scale = max(first(D.S),eps())
    nullity = count(s -> s <= rank_tol*scale,D.S)
    nullity == 1 || return (status=:fold_rank_failure, nullity=nullity,
        transversality=NaN, curvature=NaN)
    v, ell = D.V[:,end], D.U[:,end]
    h = second(v)
    a, b = dot(ell,flambda), dot(ell,h)
    transversality = abs(a)/max(norm(flambda),eps())
    curvature = abs(b)/max(norm(h),scale,norm(flambda),eps())
    status = transversality <= nondeg_tol ? :fold_transversality_failure :
             curvature <= nondeg_tol ? :fold_curvature_failure : :simple_fold
    return (;status,nullity,transversality,curvature)
end

function _cpf_verify_fold(w, point; branches=nothing, rank_tol=1e-7, nondeg_tol=1e-6)
    J = copy(_cpf_jacobian!(w,point.state;branches))
    started = w.stats === nothing ? UInt64(0) : time_ns()
    certificate = _cpf_fold_certificate(J,-w.direction,d -> _cpf_second_direction(w,point.state,d);
                                        rank_tol,nondeg_tol)
    if w.stats !== nothing
        w.stats.fold_checks += 1
        w.stats.fold_ns += time_ns()-started
    end
    return certificate
end

# A cubic Hermite guard model exposes interior minima even when both endpoints
# are feasible. It is only a sampling guide, never a solution or an enclosure.
function _cpf_hermite_minima(f0,f1,d0,d1,h)
    a = 2f0-2f1+h*(d0+d1)
    b = -3f0+3f1-h*(2d0+d1)
    c = h*d0
    roots = Float64[]
    if abs(a) <= eps()*max(abs(b),abs(c),1.)
        b > 0 && push!(roots,-c/(2b))
    else
        disc = 4b*b-12a*c
        if disc >= 0
            for u in ((-2b-sqrt(disc))/(6a),(-2b+sqrt(disc))/(6a))
                6a*u+2b > 0 && push!(roots,u)
            end
        end
    end
    return filter(u -> 0 < u < 1,roots)
end

# Sample corrected points in the SAME local section coordinate used by the
# step corrector. Tangent derivatives must be divided by row*t, not treated
# as derivatives with respect to this coordinate without reparameterization.
# Mandatory subdivision plus Hermite stationary-point probes detects ordinary
# exit/reentry pairs. This is a numerical safeguard, not interval certification
# of arbitrary oscillations between samples.
function _cpf_scan_controls(w, previous, candidate, step, guards, branches, weights;
                            tol,event_tol,max_newton, subdivisions=8)
    z0 = vcat(previous.state,previous.lambda)
    row = weights.*previous.tangent
    samples = Dict(0.0 => previous,Float64(step) => candidate)
    function sample(s)
        haskey(samples,s) && return samples[s]
        sol = _cpf_correct(w,z0+s*previous.tangent,row;tol,max_newton,branches)
        sol === nothing && return nothing
        z,it,residual = sol
        p = _cpf_point(w,z,previous.tangent,weights,it,residual;branches)
        if p !== nothing
            dot(row,p.tangent) > .1 || return nothing
            samples[s] = p
        end
        return p
    end
    grid = collect(range(0.,step;length=subdivisions+1))
    for s in grid
        sample(s) === nothing && return (status=:event_scan_failure,hits=[])
    end
    for i in 1:subdivisions
        a,b = grid[i],grid[i+1]
        left,right = samples[a],samples[b]
        dl,dr = dot(row,left.tangent),dot(row,right.tangent)
        min(dl,dr) > .1 || return (status=:event_scan_parameterization_failure,hits=[])
        for g in guards
            f0,f1 = _cpf_margin(w,left.state,g,branches),_cpf_margin(w,right.state,g,branches)
            d0 = _cpf_margin_derivative(w,left.tangent,g,branches)/dl
            d1 = _cpf_margin_derivative(w,right.tangent,g,branches)/dr
            for u in _cpf_hermite_minima(f0,f1,d0,d1,b-a)
                sample(a+u*(b-a)) === nothing && return (status=:event_scan_failure,hits=[])
            end
            # Refine a bracketed minimum using corrected derivative evaluations.
            if d0 < 0 < d1
                lo,hi = a,b
                for _ in 1:40
                    s = (lo+hi)/2
                    p = sample(s)
                    p === nothing && return (status=:event_scan_failure,hits=[])
                    derivative = _cpf_margin_derivative(w,p.tangent,g,branches)
                    if derivative < 0
                        lo = s
                    else
                        hi = s
                    end
                    hi-lo <= tol && break
                end
            end
        end
    end
    ordered = sort!(collect(keys(samples)))
    hits = []
    grazing = []
    for g in guards
        # An intermediate sample may already be slightly infeasible, but not
        # negative enough to trigger an event by itself. Keep the last truly
        # feasible sample: the immediately preceding sample need not bracket
        # zero when a later sample finally exceeds the detection tolerance.
        feasible = first(ordered)
        for i in 2:length(ordered)
            b = ordered[i]
            f = _cpf_margin(w,samples[b].state,g,branches)
            if f < -event_tol
                push!(hits,(guard=g,lo=feasible,hi=b))
                break
            elseif abs(f) <= event_tol &&
                   abs(_cpf_margin_derivative(w,samples[b].tangent,g,branches)) <= event_tol
                push!(grazing,(guard=g,step=b,point=samples[b]))
            end
            f >= 0 && (feasible = b)
        end
    end
    if !isempty(grazing)
        first_grazing = grazing[argmin(getproperty.(grazing,:step))]
        first_hit = isempty(hits) ? Inf : minimum(h.lo for h in hits)
        first_grazing.step <= first_hit && return (status=:control_guard_grazing,hits=hits,
                                                   grazing=first_grazing)
    end
    return (status=:ok,hits=hits)
end
