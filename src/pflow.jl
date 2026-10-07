using FiniteDiff

function resfun!(f::AbstractArray, x::AbstractArray, vmag, vang, pinj, qinj, ybus_mat, bus_type, pq_idx, pqv_idx)
    fill!(f, 0.0)
    npq = sum(bus_type .== 1)
    nbus = length(bus_type)

    for i in 1:nbus
        if pq_idx[i] > 0
            vmag[i] = x[pq_idx[i]]
        end

        if pqv_idx[i] > 0
            vang[i] = x[npq + pqv_idx[i]]
        end
    end

    # Get rows and non-zero values
    rows = rowvals(ybus_mat)
    vals = nonzeros(ybus_mat)

    for fr in 1:nbus
        if pq_idx[fr] > 0
            f[pq_idx[fr]] -= qinj[fr]

            for i in nzrange(ybus_mat, fr)
                to = rows[i]
                val = vals[i]
                gij = real(val)
                bij = imag(val)

                angleij = vang[fr] - vang[to]

                f[pq_idx[fr]] += vmag[fr]*vmag[to]*(gij*sin(angleij) - bij*cos(angleij))
            end
        end

        if pqv_idx[fr] > 0
            f[npq + pqv_idx[fr]] -= pinj[fr]

            for i in nzrange(ybus_mat, fr)
                to = rows[i]
                val = vals[i]
                gij = real(val)
                bij = imag(val)

                angleij = vang[fr] - vang[to]

                f[npq + pqv_idx[fr]] += vmag[fr]*vmag[to]*(gij*cos(angleij) + bij*sin(angleij))
            end
        end
    end
end

function compute_jac_nnz(ybus_mat::SparseMatrixCSC{ComplexF64, Int64}, pq_idx::Vector{Int64}, pqv_idx::Vector{Int64})
    nnz = 0
    for i in 1:size(ybus_mat, 2)
        for j in nzrange(ybus_mat, i)
            to = rowvals(ybus_mat)[j]
            if pq_idx[to] > 0 && pq_idx[i] > 0
                nnz += 4
            elseif pq_idx[to] > 0 && pqv_idx[i] > 0
                nnz += 2
            elseif pqv_idx[to] > 0 && pq_idx[i] > 0
                nnz += 2
            elseif pqv_idx[to] > 0 && pqv_idx[i] > 0
                nnz += 1
            end
        end
    end
    return nnz
end
 
function fill_jacobian!(
    x::Vector{Float64}, 
    vmag::Vector{Float64}, 
    vang::Vector{Float64}, 
    pinj::Vector{Float64}, 
    qinj::Vector{Float64}, 
    ybus_mat::SparseArrays.SparseMatrixCSC{ComplexF64, Int64}, 
    bus_type::Vector{Int64}, 
    pq_idx::Vector{Int64}, 
    pqv_idx::Vector{Int64}, 
    row_jac::Vector{Int64}, 
    col_jac::Vector{Int64}, 
    val_jac::Vector{Float64}
)
    npq = sum(bus_type .== 1)
    nbus = length(bus_type)

    rows = rowvals(ybus_mat)
    vals = nonzeros(ybus_mat)
    
    for i in 1:nbus
        if pq_idx[i] > 0
            vmag[i] = x[pq_idx[i]]
        end

        if pqv_idx[i] > 0
            vang[i] = x[npq + pqv_idx[i]]
        end
    end

    ptr = 1

    for fr in 1:nbus
        if pq_idx[fr] > 0
            vmag_fr_idx = pq_idx[fr]
            vang_fr_idx = npq + pqv_idx[fr]

            bij = imag(ybus_mat[fr, fr])
            accum_self_vmag = -2*vmag[fr]*bij
            accum_self_vang = 0.0

            for i in nzrange(ybus_mat, fr)
                to = rows[i]
                if to == fr
                    continue
                end
                nz_val = vals[i]
                gij = real(nz_val)
                bij = imag(nz_val)
                angleij = vang[fr] - vang[to]

                accum_self_vmag += vmag[to]*(gij*sin(angleij) - bij*cos(angleij))
                accum_self_vang += vmag[fr]*vmag[to]*(gij*cos(angleij) + bij*sin(angleij))

                if pqv_idx[to] > 0
                    vang_to_idx = npq + pqv_idx[to]
                    row_jac[ptr] = pq_idx[fr]
                    col_jac[ptr] = vang_to_idx
                    val_jac[ptr] = vmag[fr]*vmag[to]*(-gij*cos(angleij) - bij*sin(angleij))
                    ptr += 1
                end

                if pq_idx[to] > 0
                    vmag_to_idx = pq_idx[to]
                    row_jac[ptr] = pq_idx[fr]
                    col_jac[ptr] = vmag_to_idx
                    val_jac[ptr] = vmag[fr]*(gij*sin(angleij) - bij*cos(angleij))
                    ptr += 1
                end
            end

            row_jac[ptr] = pq_idx[fr]
            col_jac[ptr] = vmag_fr_idx
            val_jac[ptr] = accum_self_vmag
            ptr += 1

            row_jac[ptr] = pq_idx[fr]
            col_jac[ptr] = vang_fr_idx
            val_jac[ptr] = accum_self_vang
            ptr += 1
        end

        if pqv_idx[fr] > 0
            gij = real(ybus_mat[fr, fr])
            accum_self_vmag = 2*vmag[fr]*gij
            accum_self_vang = 0.0

            for i in nzrange(ybus_mat, fr)
                to = rows[i]
                if to == fr
                    continue
                end
                nz_val = vals[i]
                gij = real(nz_val)
                bij = imag(nz_val)
                angleij = vang[fr] - vang[to]

                accum_self_vmag += vmag[to]*(gij*cos(angleij) + bij*sin(angleij))
                accum_self_vang += vmag[fr]*vmag[to]*(-gij*sin(angleij) + bij*cos(angleij))

                if pqv_idx[to] > 0
                    vang_to_idx = npq + pqv_idx[to]
                    row_jac[ptr] = npq + pqv_idx[fr]
                    col_jac[ptr] = vang_to_idx
                    val_jac[ptr] = vmag[fr]*vmag[to]*(gij*sin(angleij) - bij*cos(angleij))
                    ptr += 1
                end

                if pq_idx[to] > 0
                    vmag_to_idx = pq_idx[to]
                    row_jac[ptr] = npq + pqv_idx[fr]
                    col_jac[ptr] = vmag_to_idx
                    val_jac[ptr] = vmag[fr]*(gij*cos(angleij) + bij*sin(angleij))
                    ptr += 1
                end
            end

            if pq_idx[fr] > 0
                vmag_fr_idx = pq_idx[fr]
                row_jac[ptr] = npq + pqv_idx[fr]
                col_jac[ptr] = vmag_fr_idx
                val_jac[ptr] = accum_self_vmag
                ptr += 1
            end
            vang_fr_idx = npq + pqv_idx[fr]
            row_jac[ptr] = npq + pqv_idx[fr]
            col_jac[ptr] = vang_fr_idx
            val_jac[ptr] = accum_self_vang
            ptr += 1
        end
    end
end

function construct_jacobian(
    x::Vector{Float64}, 
    vmag::Vector{Float64}, 
    vang::Vector{Float64}, 
    pinj::Vector{Float64}, 
    qinj::Vector{Float64}, 
    ybus_mat::SparseArrays.SparseMatrixCSC{ComplexF64, Int64}, 
    bus_type::Vector{Int64}, 
    pq_idx::Vector{Int64}, 
    pqv_idx::Vector{Int64}
)::SparseArrays.SparseMatrixCSC{Float64, Int64}
    nnz = compute_jac_nnz(ybus_mat, pq_idx, pqv_idx)

    row_jac = zeros(Int64, nnz)
    col_jac = zeros(Int64, nnz)
    val_jac = zeros(Float64, nnz)

    fill_jacobian!(x, vmag, vang, pinj, qinj, ybus_mat, bus_type, pq_idx, pqv_idx, row_jac, col_jac, val_jac)
    return sparse(row_jac, col_jac, val_jac)
end

function update_jacobian!(
    jac_mat::SparseArrays.SparseMatrixCSC{Float64, Int64}, 
    x::Vector{Float64}, 
    vmag::Vector{Float64}, 
    vang::Vector{Float64}, 
    pinj::Vector{Float64}, 
    qinj::Vector{Float64}, 
    ybus_mat::SparseArrays.SparseMatrixCSC{ComplexF64, Int64}, 
    bus_type::Vector{Int64}, 
    pq_idx::Vector{Int64}, 
    pqv_idx::Vector{Int64}
)
    # Rebuild sparsity structure from scratch (Jacobian pattern may change between solves)
    row_jac = zeros(Int64, length(jac_mat.nzval))
    col_jac = zeros(Int64, length(jac_mat.nzval))
    val_jac = zeros(Float64, length(jac_mat.nzval))

    fill_jacobian!(x, vmag, vang, pinj, qinj, ybus_mat, bus_type, pq_idx, pqv_idx, row_jac, col_jac, val_jac)
    J = sparse(row_jac, col_jac, val_jac)
    jac_mat.rowval .= J.rowval
    jac_mat.colptr .= J.colptr
    jac_mat.nzval .= J.nzval
end

function compute_pinj!(sinj, v, ybus_mat, nbus)
    rows = rowvals(ybus_mat)
    vals = nonzeros(ybus_mat)

    for fr_bus in 1:nbus

        sinj[2*fr_bus-1] = 0.0 # P
        sinj[2*fr_bus] = 0.0 # Q

        vmag_i = v[2*fr_bus-1]
        vang_i = v[2*fr_bus]
        angleij = 0.0

        for i in nzrange(ybus_mat, fr_bus)
            val = vals[i]
            gij = real(val)
            bij = imag(val)

            to_bus = rows[i]

            vmag_j = v[2*to_bus-1]
            vang_j = v[2*to_bus]

            angleij = vang_i - vang_j

            sinj[2*fr_bus-1] += vmag_i*vmag_j*(gij*cos(angleij)
                + bij*sin(angleij))

            sinj[2*fr_bus] += vmag_i*vmag_j*(gij*sin(angleij)
                - bij*cos(angleij))
        end
    end
end

function runpf(psys::PowerSystem; verbose=false, fdiff=false)

    prF = psys.profiler

    bus_type = [bus.type for bus in psys.buses]
    vmag = [bus.v0m for bus in psys.buses]
    vang = [bus.v0a for bus in psys.buses]
    pinj = zeros(Float64, length(psys.buses))
    qinj = zeros(Float64, length(psys.buses))

    for gen in psys.gens
        pinj[gen.bus] += gen.psch
        qinj[gen.bus] += gen.qsch
    end

    for load in psys.loads
        pinj[load.bus] -= load.pd
        qinj[load.bus] += load.qd
    end

    nslack = sum(bus_type .== 3)
    npv = sum(bus_type .== 2)
    npq = sum(bus_type .== 1)
    nbuses = length(bus_type)

    x0 = zeros(2*npq + npv)

    pq_bus = bus_type .== 1
    pq_idx = cumsum(pq_bus) .* pq_bus

    pqv_bus = (bus_type .== 1) .+ (bus_type .== 2)
    pqv_idx = cumsum(pqv_bus) .* pqv_bus

    for (idx, bus) in enumerate(psys.buses)
        if pq_idx[idx] > 0
            x0[pq_idx[idx]] = bus.v0m
        end

        if pqv_idx[idx] > 0
            x0[npq + pqv_idx[idx]] = bus.v0a
        end
    end

    # NLsolve set up
    function func!(f::AbstractArray, x::AbstractArray)
        @timeit prF "pflow: resfun" resfun!(f, x, vmag, vang, pinj, qinj, psys.network.ybus, bus_type, pq_idx, pqv_idx)
    end

    function jac!(jac::AbstractArray, x::AbstractArray)
        @timeit prF "pflow: update_jac" update_jacobian!(jac, x, vmag, vang, pinj, qinj, psys.network.ybus, bus_type, pq_idx, pqv_idx)
    end

    if fdiff
        @timeit prF "pflow: nlsolve - fdiff" result = nlsolve(func!, x0, method=:newton, iterations=50, show_trace=verbose)
    else
        J0 = construct_jacobian(x0, vmag, vang, pinj, qinj, psys.network.ybus, bus_type, pq_idx, pqv_idx)
        f0 = zero(x0)
        df = OnceDifferentiable(func!, jac!, x0, f0, J0)
        @timeit prF "pflow: nlsolve - jac" result = nlsolve(df, x0, method=:newton, iterations=50, show_trace=verbose)
        # Alternative: custom Newton solver (currently using NLsolve)
        #@timeit prF "pflow: newton" result, success = newton(x0, J0, func!, jac!, tol = 1e-8, verbose = true)
    end

    # Retrieve solution
    sol = result.zero

    # Retrieve voltage magnitudes and angles
    for i in 1:nbuses
        if pq_idx[i] > 0
            vmag[i] = sol[pq_idx[i]]
        end

        if pqv_idx[i] > 0
            vang[i] = sol[npq + pqv_idx[i]]
        end
    end

    # We will return a vector v and pinj such that
    # v = [vmag1, vang1, vmag2, vang2, ...]
    # Sinj = [pinj1, qinj1, pinj2, qinj2, ...]
    v = Array(vec([vmag vang]'))
    sinj = zeros(length(v))
    compute_pinj!(sinj, v, psys.network.ybus, nbuses)
    psol = PowerFlowSolution(v, sinj)
    return psol
end

# Apply a converged power-flow solution back onto the PowerSystem: bus
# voltages, and the reactive (and, at the slack, active) dispatch implied by
# the solution. Buses that the Q-limit loop has pinned are type 1 (PQ) by then,
# so their `qsch` is left at the limit value rather than recomputed.
# Split a bus's reactive dispatch `qtot` across its generators.
#
# The even split is what this code has always done and is what you want with no
# limit data: every machine at the bus carries the same share. But under
# Q-limit enforcement it is wrong at the machine level. A bus can sit inside
# its AGGREGATE limits while an even split pushes a small unit far outside its
# own: ACTIVSg2000 bus 7422 has one 3.5 MVAr unit alongside four 57 MVAr units,
# and an even split hands the small one 0.41 pu against a 0.035 pu ceiling.
#
# So when limits are being enforced we project the even split onto the box
# while preserving the bus total, which is what uqgrid's power flow does:
# hand out an equal share; any unit for which that share is outside its own
# range is fixed at the violated bound and removed; re-share the remainder
# among the rest; repeat. Each pass fixes at least one unit, so it terminates.
#
# This is deliberately a strict generalisation of the even split rather than,
# say, allocating in proportion to each unit's reactive range: when no limit
# binds it reproduces the even split exactly, so enabling `enforce_q_limits`
# changes the dispatch only at buses where a machine really is at a bound.
#
# `qtot` outside the aggregate box has no projection that preserves the sum.
# That happens only at the slack, which is intentionally never limited, so we
# fall back to the even split there.
function _distribute_q!(psys::PowerSystem, gidx, qtot::Float64, proportional::Bool)
    n = length(gidx)
    even!() = for g in gidx; psys.gens[g].qsch = qtot / n; end

    proportional || return even!()
    qmin_s = sum(psys.gens[g].qmin for g in gidx)
    qmax_s = sum(psys.gens[g].qmax for g in gidx)
    (qtot < qmin_s || qtot > qmax_s) && return even!()

    free = trues(n)
    remaining = qtot
    nfree = n
    while nfree > 0
        share = remaining / nfree
        fixed = 0
        for k in 1:n
            free[k] || continue
            gen = psys.gens[gidx[k]]
            if share < gen.qmin
                gen.qsch = gen.qmin
            elseif share > gen.qmax
                gen.qsch = gen.qmax
            else
                continue
            end
            remaining -= gen.qsch
            free[k] = false
            nfree -= 1
            fixed += 1
        end
        if fixed == 0                       # share is feasible for every unit left
            for k in 1:n
                free[k] && (psys.gens[gidx[k]].qsch = share)
            end
            break
        end
    end
    return nothing
end

function _apply_pf_solution!(psys::PowerSystem, psol::PowerFlowSolution, bus_to_gen;
                             proportional_q::Bool=false)
    # Update bus voltages
    for (idx, bus) in enumerate(psys.buses)
        bus.v0m = psol.volt[2*idx-1]
        bus.v0a = psol.volt[2*idx]
    end

    # compute generation vector
    sgen = copy(psol.sinj)
    for (idx, load) in enumerate(psys.loads)
        sgen[2*load.bus-1] += load.pd
        sgen[2*load.bus] -= load.qd
    end

    # for all the PV buses. we distribute the reactive power
    # among all the generators evenly.
    for (idx, bus) in enumerate(psys.buses)
        gidx = bus_to_gen[idx]
        isempty(gidx) && continue
        if bus.type == 2
            _distribute_q!(psys, gidx, sgen[2*idx], proportional_q)
        elseif bus.type == 3
            for gen_idx in gidx
                psys.gens[gen_idx].psch = sgen[2*idx-1] / length(gidx)
            end
            _distribute_q!(psys, gidx, sgen[2*idx], proportional_q)
        end
    end
    return sgen
end

# bus number to generator. This is an array of arrays where bus_to_gen[i] gives
# an array with the indices of all generators connected to bus i.
# NOTE: might need to create this structure in other place
# and store it in the PowerSystem struct
function _bus_to_gen(psys::PowerSystem)
    bus_to_gen = [Array{Int}(undef, 0) for i in 1:length(psys.buses)]
    for (idx, gen) in enumerate(psys.gens)
        push!(bus_to_gen[gen.bus], idx)
    end
    return bus_to_gen
end

"""
    runpf!(psys; verbose=false, fdiff=false, enforce_q_limits=false,
           max_q_outer=50, q_tol=1e-6)

Same as `runpf` but writes the solution back into the `PowerSystem` in place.

## Generator reactive limits (`enforce_q_limits`)

With `enforce_q_limits=false` (the default, and the behaviour before this
option existed) a PV bus holds its scheduled voltage no matter how much
reactive power that takes. The solution is then a fixed point of the power-flow
equations but not necessarily a physical operating point: machines can be left
absorbing or producing far outside their nameplate `QT`/`QB`.

That is not a cosmetic problem. On ACTIVSg2000, 200 of 432 generators solve
outside their limits (one bus absorbs 279 MVAr against an 8 MVAr floor). The
excess reactive flow drags 20 machines past their pull-out angle -- internal
angle `delta - angle(V)` above 90 degrees, where `dP/ddelta < 0` -- which puts
18 non-oscillatory unstable eigenvalues into the linearised dynamics. Any
transient run from that point diverges regardless of the disturbance.

With `enforce_q_limits=true` we run the standard outer loop:

 1. Solve the power flow with the current PV/PQ assignment.
 2. For each bus that started as PV, compare the reactive dispatch against the
    sum of its generators' limits. If it is above the aggregate `qmax` (below
    `qmin`), convert the bus to PQ, hold every generator there at its own
    `qmax` (`qmin`), and release the voltage.
 3. Repeat until an iteration converts nothing, or `max_q_outer` is reached.

Pinning each generator at its own limit is exact rather than a heuristic: a bus
sits at its aggregate limit precisely when every machine on it is at its own.

Conversion is one-way, as in MATPOWER's default `enforce_q_lims = 1`. Allowing
a pinned bus back to PV once its voltage recovers past setpoint is tempting and
is what `enforce_q_lims = 2` does, but on ACTIVSg2000 it cycles: a handful of
buses pin, release, and re-pin forever, and the loop exits on the iteration cap
with those buses still violating. One-way conversion is monotone -- every
iteration either pins at least one more bus or stops -- so it terminates, and
when it stops no PV bus is outside its limits. The cost is mild conservatism: a
bus can stay pinned when it could have held its voltage after the rest of the
system moved.

The slack bus is never limited -- it has to absorb the system mismatch, and
capping it would leave the power flow with no degree of freedom to close on.
If the slack ends up outside its limits that is a property of the case, and
`verbose=true` reports it.

Buses whose only generation is a `StaticGenerator` stub are switched here too,
but note the stub regulates voltage with unlimited reactive power *during the
transient* regardless -- the limit is respected by the operating point, not by
the dynamics.

Limits come from the `.raw` generator records (`QT`/`QB`, converted to the
system base at parse time). Records that carry the usual +/-9999 placeholder
simply never bind. Cases built without limit data default to `+/-Inf`, so
`enforce_q_limits=true` is a no-op for them rather than an error.

Returns the number of buses left pinned at a limit.
"""
function runpf!(psys::PowerSystem; verbose=false, fdiff=false,
                enforce_q_limits::Bool=false, max_q_outer::Int=50,
                q_tol::Float64=1e-6)
    bus_to_gen = _bus_to_gen(psys)

    if !enforce_q_limits
        _apply_pf_solution!(psys, runpf(psys; verbose=verbose, fdiff=fdiff), bus_to_gen)
        return 0
    end

    nbus = length(psys.buses)
    orig_type = [bus.type for bus in psys.buses]
    pinned    = zeros(Int, nbus)   # 0 free, +1 held at qmax, -1 held at qmin

    outer = 0
    for it in 1:max_q_outer
        outer = it
        _apply_pf_solution!(psys, runpf(psys; verbose=verbose, fdiff=fdiff), bus_to_gen;
                            proportional_q=true)

        changed = 0
        for i in 1:nbus
            orig_type[i] == 2 || continue          # slack and pure PQ are exempt
            gidx = bus_to_gen[i]
            isempty(gidx) && continue

            qmax = sum(psys.gens[g].qmax for g in gidx)
            qmin = sum(psys.gens[g].qmin for g in gidx)
            qmax >= qmin || continue               # inconsistent limit data; skip

            pinned[i] == 0 || continue             # one-way: never un-pin
            q = sum(psys.gens[g].qsch for g in gidx)
            if q > qmax + q_tol
                pinned[i] = 1
                psys.buses[i].type = 1
                for g in gidx
                    psys.gens[g].qsch = psys.gens[g].qmax
                end
                changed += 1
            elseif q < qmin - q_tol
                pinned[i] = -1
                psys.buses[i].type = 1
                for g in gidx
                    psys.gens[g].qsch = psys.gens[g].qmin
                end
                changed += 1
            end
        end

        verbose && @info "pflow Q-limit outer iteration $it: $changed switch(es), $(count(!=(0), pinned)) bus(es) pinned"
        changed == 0 && break
    end

    npin = count(!=(0), pinned)
    if outer == max_q_outer && npin > 0
        # We stopped on the iteration cap, so the last solve may still violate.
        still = 0
        for i in 1:nbus
            orig_type[i] == 2 && pinned[i] == 0 || continue
            gidx = bus_to_gen[i]; isempty(gidx) && continue
            q = sum(psys.gens[g].qsch for g in gidx)
            (q > sum(psys.gens[g].qmax for g in gidx) + q_tol ||
             q < sum(psys.gens[g].qmin for g in gidx) - q_tol) && (still += 1)
        end
        still > 0 && @warn "pflow Q-limit loop hit max_q_outer=$max_q_outer with $still bus(es) still violating"
    end

    if verbose
        sl = findfirst(==(3), orig_type)
        if sl !== nothing && !isempty(bus_to_gen[sl])
            q = sum(psys.gens[g].qsch for g in bus_to_gen[sl])
            qm = sum(psys.gens[g].qmax for g in bus_to_gen[sl])
            qb = sum(psys.gens[g].qmin for g in bus_to_gen[sl])
            (q > qm + q_tol || q < qb - q_tol) &&
                @info "pflow: slack bus is outside its reactive limits (q=$q, [$qb, $qm]); slack is never limited"
        end
    end
    return npin
end
