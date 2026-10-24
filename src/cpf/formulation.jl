"""
    LoadingDirection(ps; dp, dq)
    LoadingDirection(ps)

Additive net-injection direction in per-unit on the system MVA base:
`S_specified(λ) = S_base + λ * (dp + im*dq)`. Positive entries increase
injection, not demand. Vectors use **internal bus indices** and are copied.
The reference-bus entries must be zero (its dispatch balances the network).

With no keywords, scale the existing constant-power loads, leaving reference
bus demand unchanged. The package stores reactive load demand as `-load.qd`;
`dq` here always uses the physical reactive-injection sign.
"""
struct LoadingDirection
    dp::Vector{Float64}
    dq::Vector{Float64}
    bus_numbers::Vector{Int}
    function LoadingDirection(ps::PowerSystem; dp=nothing, dq=nothing)
        n = length(ps.buses)
        if dp === nothing && dq === nothing
            p = zeros(n)
            q = zeros(n)
            for load in ps.loads
                ps.buses[load.bus].type == 3 && continue
                p[load.bus] -= load.pd
                q[load.bus] += load.qd
            end
        else
            p = dp === nothing ? zeros(n) : Float64.(dp)
            q = dq === nothing ? zeros(n) : Float64.(dq)
        end
        length(p) == length(q) == n || throw(ArgumentError("direction needs one entry per bus"))
        all(isfinite, p) && all(isfinite, q) || throw(ArgumentError("direction must be finite"))
        for (i, bus) in enumerate(ps.buses)
            bus.type == 3 && (p[i] != 0 || q[i] != 0) &&
                throw(ArgumentError("reference-bus direction must be zero"))
        end
        new(copy(p), copy(q), [b.i for b in ps.buses])
    end
end

# Workspace owns all data; constructing or evaluating it never modifies ps.
struct CPFWorkspace
    ybus::SparseMatrixCSC{ComplexF64,Int}
    buses::Vector{Int}
    active::Vector{Int}
    index::Vector{Int}
    controlled::Vector{Int}
    pbase::Vector{Float64}
    qbase::Vector{Float64}
    vset::Vector{Float64}
    angle::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    direction::Vector{Float64}
    initial::Vector{Float64}
    limits::Symbol
    control_scale::Float64
    jacobian::SparseMatrixCSC{Float64,Int}
    slots::Dict{Tuple{Int,Int},Int}
    stats::Union{Nothing,ContinuationStats}
end

function _cpf_workspace(ps::PowerSystem, direction::LoadingDirection;
                        limits::Symbol=:complementarity, control_scale::Real=1.0,
                        stats::Union{Nothing,ContinuationStats}=nothing)
    started = stats === nothing ? UInt64(0) : time_ns()
    limits in (:none, :complementarity) || throw(ArgumentError("limits must be :none or :complementarity"))
    isfinite(control_scale) && control_scale > 0 || throw(ArgumentError("control_scale must be finite and positive"))
    n = length(ps.buses)
    n >= 2 || throw(ArgumentError("continuation requires at least two buses"))
    direction.bus_numbers == [b.i for b in ps.buses] || throw(ArgumentError("direction bus ordering differs from system"))
    length(direction.dp) == length(direction.dq) == n || throw(ArgumentError("invalid direction lengths"))
    all(isfinite, direction.dp) && all(isfinite, direction.dq) || throw(ArgumentError("direction must be finite"))
    count(b -> b.type == 3, ps.buses) == 1 || throw(ArgumentError("exactly one reference bus is supported"))
    all(b -> b.type in (1, 2, 3), ps.buses) || throw(ArgumentError("unsupported bus type"))
    all(b -> isfinite(b.v0m) && b.v0m > 0 && isfinite(b.v0a), ps.buses) ||
        throw(ArgumentError("initial voltages must be finite with positive magnitudes"))
    ref = findfirst(b -> b.type == 3, ps.buses)
    direction.dp[ref] == direction.dq[ref] == 0 || throw(ArgumentError("reference-bus direction must be zero"))
    # Build from current topology rather than trusting a possibly stale network.
    ybus = create_ybus_complex(ps)
    seen = falses(n)
    seen[ref] = true
    queue = [ref]
    for j in queue
        for k in nzrange(ybus, j)
            i = rowvals(ybus)[k]
            i == j && continue
            iszero(nonzeros(ybus)[k]) && continue
            if !seen[i]
                seen[i] = true
                push!(queue, i)
            end
        end
    end
    all(seen) || throw(ArgumentError("disconnected networks are not supported"))
    active = findall(b -> b.type != 3, ps.buses)
    controlled = findall(b -> b.type == 2, ps.buses)
    m = length(active)
    nc = length(controlled)
    dim = 2m + nc
    index = zeros(Int, n)
    index[active] = 1:m
    pbase, qbase, qgen = zeros(n), zeros(n), zeros(n)
    lower, upper = zeros(nc), zeros(nc)
    for gen in ps.gens
        gen.status || continue
        isfinite(gen.psch) && isfinite(gen.qsch) || throw(ArgumentError("generator dispatch must be finite"))
        pbase[gen.bus] += gen.psch
        qgen[gen.bus] += gen.qsch
    end
    for load in ps.loads
        isfinite(load.pd) && isfinite(load.qd) || throw(ArgumentError("loads must be finite"))
        pbase[load.bus] -= load.pd
        qbase[load.bus] += load.qd
    end
    qbase .+= qgen
    for (k, bus) in enumerate(controlled)
        gens = filter(g -> g.status && g.bus == bus, ps.gens)
        isempty(gens) && throw(ArgumentError("voltage-controlled bus $(ps.buses[bus].i) has no online generator"))
        all(g -> !isnan(g.qmin) && !isnan(g.qmax) && g.qmin <= g.qmax &&
                 g.qmin != Inf && g.qmax != -Inf, gens) || throw(ArgumentError("invalid reactive limits"))
        lower[k] = sum(g.qmin for g in gens)
        upper[k] = sum(g.qmax for g in gens)
        qbase[bus] -= qgen[bus]
    end
    initial = vcat([ps.buses[i].v0a for i in active],
                   [ps.buses[i].v0m for i in active], qgen[controlled])
    b = vcat(direction.dp[active], direction.dq[active], zeros(nc))
    # Union pattern includes both possible control derivatives, even when zero.
    entries = Set{Tuple{Int,Int}}()
    for j in 1:n, k in nzrange(ybus, j)
        i = rowvals(ybus)[k]
        ii, jj = index[i], index[j]
        ii == 0 && continue
        jj == 0 && continue
        for row in (ii, m + ii), col in (jj, m + jj)
            push!(entries, (row, col))
        end
    end
    for i in 1:m, row in (i, m + i), col in (i, m + i)
        push!(entries, (row, col))
    end
    for (k, bus) in enumerate(controlled)
        i = index[bus]
        push!(entries, (m + i, 2m + k))
        push!(entries, (2m + k, m + i))
        push!(entries, (2m + k, 2m + k))
    end
    pairs = collect(entries)
    J = sparse(first.(pairs), last.(pairs), ones(length(pairs)), dim, dim)
    slots = Dict((rowvals(J)[k], j) => k for j in 1:dim for k in nzrange(J, j))
    workspace = CPFWorkspace(ybus, [b.i for b in ps.buses], active, index, controlled,
                        pbase, qbase, [b.v0m for b in ps.buses], [b.v0a for b in ps.buses],
                        lower, upper, b, initial, limits, Float64(control_scale), J, slots, stats)
    if stats !== nothing
        stats.setups += 1
        stats.setup_ns += time_ns()-started
    end
    return workspace
end

# Projection residual with positive Q/voltage scaling. At an exact kink choose
# the saturated derivative; an event locator must obtain one-sided tangents.
function _cpf_control(q, v, setpoint, lower, upper, scale, limits)
    limits === :none && return (scale * (v - setpoint), 0.0, scale)
    trial = q - scale * (v - setpoint)
    if trial <= lower
        return (q - lower, 1.0, 0.0)
    elseif trial >= upper
        return (q - upper, 1.0, 0.0)
    end
    return (scale * (v - setpoint), 0.0, scale)
end

function _cpf_voltages(w::CPFWorkspace, x)
    mag, angle = copy(w.vset), copy(w.angle)
    m = length(w.active)
    mag[w.active] .= @view x[m+1:2m]
    angle[w.active] .= @view x[1:m]
    return mag, angle
end

# A branch override extends one smooth control regime beyond its feasible
# region solely for event bracketing. Accepted points still satisfy the exact
# projection residual; a bound change never changes the workspace sparsity.
function _cpf_control_branch(w, k, q, v, branches)
    branches === nothing && return _cpf_control(q, v, w.vset[w.controlled[k]],
        w.lower[k], w.upper[k], w.control_scale, w.limits)
    branch = branches[k]
    branch === :free && return (w.control_scale*(v-w.vset[w.controlled[k]]), 0., w.control_scale)
    branch in (:lower, :fixed) && return (q-w.lower[k], 1., 0.)
    branch === :upper && return (q-w.upper[k], 1., 0.)
    throw(ArgumentError("invalid reactive-control branch"))
end

function _cpf_residual!(f, w::CPFWorkspace, x, lambda; branches=nothing)
    started = w.stats === nothing ? UInt64(0) : time_ns()
    m = length(w.active)
    mag, angle = _cpf_voltages(w, x)
    v = mag .* cis.(angle)
    s = v .* conj.(w.ybus * v)
    for (i, bus) in enumerate(w.active)
        f[i] = real(s[bus]) - w.pbase[bus] - lambda * w.direction[i]
        f[m+i] = imag(s[bus]) - w.qbase[bus] - lambda * w.direction[m+i]
    end
    for (k, bus) in enumerate(w.controlled)
        q = x[2m+k]
        f[m+w.index[bus]] -= q
        f[2m+k] = first(_cpf_control_branch(w, k, q, mag[bus], branches))
    end
    if w.stats !== nothing
        w.stats.residuals += 1
        w.stats.residual_ns += time_ns()-started
    end
    return f
end

function _cpf_jacobian!(w::CPFWorkspace, x; branches=nothing)
    started = w.stats === nothing ? UInt64(0) : time_ns()
    J = w.jacobian
    fill!(nonzeros(J), 0.0)
    m = length(w.active)
    mag, angle = _cpf_voltages(w, x)
    add!(i, j, a) = (nonzeros(J)[w.slots[(i, j)]] += a)
    # Iterate actual (row i, column j) entries; transformer shifts can make Y
    # nonsymmetric. Diagonal angle derivatives are sums of off-diagonal terms.
    for j in axes(w.ybus, 2), k in nzrange(w.ybus, j)
        i = rowvals(w.ybus)[k]
        ii, jj = w.index[i], w.index[j]
        ii == 0 && continue
        g, b = reim(nonzeros(w.ybus)[k])
        if i == j
            add!(ii, m+ii, 2mag[i]*g)
            add!(m+ii, m+ii, -2mag[i]*b)
            continue
        end
        sn, cs = sincos(angle[i] - angle[j])
        a, c = g*cs + b*sn, g*sn - b*cs
        add!(ii, ii, -mag[i]*mag[j]*c)
        add!(m+ii, ii, mag[i]*mag[j]*a)
        add!(ii, m+ii, mag[j]*a)
        add!(m+ii, m+ii, mag[j]*c)
        if jj != 0
            add!(ii, jj, mag[i]*mag[j]*c)
            add!(m+ii, jj, -mag[i]*mag[j]*a)
            add!(ii, m+jj, mag[i]*a)
            add!(m+ii, m+jj, mag[i]*c)
        end
    end
    for (k, bus) in enumerate(w.controlled)
        _, dq, dv = _cpf_control_branch(w, k, x[2m+k], mag[bus], branches)
        add!(m+w.index[bus], 2m+k, -1.0)
        add!(2m+k, 2m+k, dq)
        add!(2m+k, m+w.index[bus], dv)
    end
    if w.stats !== nothing
        w.stats.jacobians += 1
        w.stats.jacobian_ns += time_ns()-started
    end
    return J
end
