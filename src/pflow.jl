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

# `v_init` (interleaved [vm1, va1, vm2, va2, ...], e.g. a previous
# `PowerFlowSolution.volt`) warm-starts Newton: PQ magnitudes and all angles
# start from it instead of the case's stored voltages. PV/slack magnitudes are
# setpoints and always come from `psys.buses`.
function runpf(psys::PowerSystem; verbose=false, fdiff=false, bus_type=nothing,
               v_init=nothing)

    prF = psys.profiler

    # `bus_type` lets runpf! re-solve with PV buses switched to PQ when their
    # generators hit a reactive limit, without mutating `psys.buses[i].type`
    # (which the dynamic model and the dataset export still read).
    bus_type = bus_type === nothing ? [bus.type for bus in psys.buses] : bus_type
    vmag = [bus.v0m for bus in psys.buses]
    vang = [bus.v0a for bus in psys.buses]
    if v_init !== nothing
        for i in eachindex(vmag)
            bus_type[i] == 1 && (vmag[i] = v_init[2i-1])
            vang[i] = v_init[2i]
        end
    end
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
            x0[pq_idx[idx]] = vmag[idx]
        end

        if pqv_idx[idx] > 0
            x0[npq + pqv_idx[idx]] = vang[idx]
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

    ok = converged(result)
    ok || verbose &&
        @warn "runpf: Newton did not converge in $(result.iterations) iterations (residual $(result.residual_norm))"

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
    psol = PowerFlowSolution(v, sinj, ok)
    return psol
end

# Generation per bus from a power-flow solution: sinj = generation - load, so
# generation = sinj + load. The injection was built as `pinj -= pd` /
# `qinj += qd` (qd is stored negated), so the inverse is `+= pd` / `-= qd`.
# (The P sign used to be `-=`, which double-counted the load whenever the
# slack bus carries load; IEEE39's slack came out at -0.855 pu, not +7.145.)
function _pf_generation(psys::PowerSystem, psol::PowerFlowSolution)
    sgen = copy(psol.sinj)
    for load in psys.loads
        sgen[2*load.bus-1] += load.pd
        sgen[2*load.bus]   -= load.qd
    end
    return sgen
end

# Share a bus's total reactive output among its generators in proportion to
# each unit's reactive range, as MATPOWER does, so every unit sits at the same
# fraction of its range and a bus at its limit puts every unit at its own
# limit. Falls back to an even split when the range is unbounded or zero
# (e.g. MATPOWER/test cases with no Q limits), preserving the old behaviour.
function _split_q!(gens::Vector{Gen}, idxs::Vector{Int}, qtot::Float64)
    isempty(idxs) && return nothing
    qmin = sum(gens[g].qmin for g in idxs)
    qmax = sum(gens[g].qmax for g in idxs)
    rng = qmax - qmin
    if isfinite(rng) && rng > 1e-12
        for g in idxs
            gi = gens[g]
            gi.qsch = gi.qmin + (qtot - qmin) * (gi.qmax - gi.qmin) / rng
        end
    else
        for g in idxs
            gens[g].qsch = qtot / length(idxs)
        end
    end
    return nothing
end

"""
    runpf!(psys; verbose=false, fdiff=false, enforce_qlims=true,
           qlim_tol=1e-6, max_qlim_iter=50)

Solve the power flow and write the solution back into `psys` (bus voltages,
generator P/Q setpoints). Returns `true` if Newton converged.
"""
function runpf!(psys::PowerSystem; verbose=false, fdiff=false,
                enforce_qlims::Bool=true, qlim_tol::Float64=1e-6,
                max_qlim_iter::Int=50)
    nbus = length(psys.buses)
    bus_to_gen = [Int[] for _ in 1:nbus]
    for (idx, gen) in enumerate(psys.gens)
        push!(bus_to_gen[gen.bus], idx)
    end

    bus_type = [bus.type for bus in psys.buses]
    psol = runpf(psys; verbose=verbose, fdiff=fdiff, bus_type=bus_type)

    nswitched = 0
    if enforce_qlims
        settled = false
        for it in 1:max_qlim_iter
            psol.converged || break   # nothing sensible to switch from
            sgen = _pf_generation(psys, psol)
            # (bus, violation, Q limit it is fixed at)
            viol = Tuple{Int,Float64,Bool}[]
            for i in 1:nbus
                bus_type[i] == 2 || continue
                gens = bus_to_gen[i]
                isempty(gens) && continue
                qmax = sum(psys.gens[g].qmax for g in gens)
                qmin = sum(psys.gens[g].qmin for g in gens)
                q = sgen[2*i]
                if q > qmax + qlim_tol
                    push!(viol, (i, q - qmax, true))
                elseif q < qmin - qlim_tol
                    push!(viol, (i, qmin - q, false))
                end
            end
            if isempty(viol)
                settled = true
                break
            end
            sort!(viol; by = v -> -v[2])
            k = length(viol)
            while true
                trial = copy(bus_type)
                for (i, _, at_max) in viol[1:k]
                    for g in bus_to_gen[i]
                        psys.gens[g].qsch = at_max ? psys.gens[g].qmax : psys.gens[g].qmin
                    end
                    trial[i] = 1          # Q now fixed via gen.qsch in runpf's qinj
                end
                # Warm-start from the solution just found: restarting from the
                # case's stored voltages after switching diverged on ACTIVSg2000
                # at perturbed loadings (residual ~1e5-1e6).
                ptry = runpf(psys; verbose=verbose, fdiff=fdiff, bus_type=trial,
                             v_init=psol.volt)
                if ptry.converged || k == 1
                    bus_type .= trial
                    psol = ptry
                    nswitched += k
                    break
                end
                k = cld(k, 2)
            end
        end
        settled || !psol.converged ||
            @warn "runpf!: reactive-limit switching did not settle in $max_qlim_iter iterations"
        nswitched > 0 && @info "runpf!: $nswitched generator bus(es) hit a reactive limit and were switched PV→PQ."
    end
    psol.converged || @warn "runpf!: power flow did not converge; the solution written back is not a steady state"

    # Update bus voltages
    for (idx, bus) in enumerate(psys.buses)
        bus.v0m = psol.volt[2*idx-1]
        bus.v0a = psol.volt[2*idx]
    end

    sgen = _pf_generation(psys, psol)
    for i in 1:nbus
        gens = bus_to_gen[i]
        isempty(gens) && continue
        if bus_type[i] == 2 || bus_type[i] == 3
            # PV / slack: Q is a result of the solve; share it across units.
            _split_q!(psys.gens, gens, sgen[2*i])
        end
        # Buses switched to PQ already hold each unit at its own limit.
        if bus_type[i] == 3
            for g in gens
                psys.gens[g].psch = sgen[2*i-1] / length(gens)
            end
        end
    end
    return psol.converged
end
