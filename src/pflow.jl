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
    # TODO: refactor
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

            if rows[i] == fr_bus
                sinj[2*fr_bus-1] += vmag_i*vmag_i*(gij*cos(angleij)
                    + bij*sin(angleij))

                sinj[2*fr_bus] += vmag_i*vmag_i*(gij*sin(angleij)
                    - bij*cos(angleij))
            else
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
        # TODO: better interface to select solver
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
    v = vec([vmag vang]')
    sinj = zeros(length(v))
    compute_pinj!(sinj, v, psys.network.ybus, nbuses)
    psol = PowerFlowSolution(v, sinj)

    return psol
end
