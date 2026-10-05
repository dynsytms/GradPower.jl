# IEEEG1 batched residual & Jacobian.
#
# States: 6 diff (x1 lead-lag, x2 valve servo, x3..x6 turbine stages)
#         + 1 alg (p_m). pref is init-derived (pvec slot 23).
#
# With  e = pref - K·w   and   t21 = T2/T1:
#   F[dp+0] = ((1 - t21)·e - x1) / T1
#   F[dp+1] = (x1 + t21·e - x2) / T3
#   F[dp+2] = (x2 - x3) / T4
#   F[dp+3] = (x3 - x4) / T5
#   F[dp+4] = (x4 - x5) / T6
#   F[dp+5] = (x5 - x6) / T7
#   F[ap]   = K1·x3 + K3·x4 + K5·x5 + K7·x6 - p_m
#
# Time constants arrive already floored at IEEEG1_TMIN (see fill_pvec!), so
# there is no branch on T == 0 here.

const IEEEG1_JAC_NENTRIES = 18

const J_G1_R1_x1 = 1
const J_G1_R1_w  = 2
const J_G1_R2_x1 = 3
const J_G1_R2_x2 = 4
const J_G1_R2_w  = 5
const J_G1_R3_x2 = 6
const J_G1_R3_x3 = 7
const J_G1_R4_x3 = 8
const J_G1_R4_x4 = 9
const J_G1_R5_x4 = 10
const J_G1_R5_x5 = 11
const J_G1_R6_x5 = 12
const J_G1_R6_x6 = 13
const J_G1_A_x3  = 14
const J_G1_A_x4  = 15
const J_G1_A_x5  = 16
const J_G1_A_x6  = 17
const J_G1_A_pm  = 18

function ieeeg1_preallocate!(coord_list::Vector{Vector{Int}},
                              table::IEEEG1Table, diff_dim::Int)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        w  = Int(table.w_idx[k])

        push!(coord_list[dp], dp)
        w != 0 && push!(coord_list[dp], w)

        push!(coord_list[dp + 1], dp)
        push!(coord_list[dp + 1], dp + 1)
        w != 0 && push!(coord_list[dp + 1], w)

        push!(coord_list[dp + 2], dp + 1)
        push!(coord_list[dp + 2], dp + 2)

        push!(coord_list[dp + 3], dp + 2)
        push!(coord_list[dp + 3], dp + 3)

        push!(coord_list[dp + 4], dp + 3)
        push!(coord_list[dp + 4], dp + 4)

        push!(coord_list[dp + 5], dp + 4)
        push!(coord_list[dp + 5], dp + 5)

        push!(coord_list[ap], dp + 2)
        push!(coord_list[ap], dp + 3)
        push!(coord_list[ap], dp + 4)
        push!(coord_list[ap], dp + 5)
        push!(coord_list[ap], ap)
    end
    return nothing
end

function ieeeg1_jac_positions!(table::IEEEG1Table, J::SparseMatrixCSC,
                                diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        w  = Int(table.w_idx[k])

        table.jac_pos[k, J_G1_R1_x1] = _find_pos(J, rows, dp, dp)
        table.jac_pos[k, J_G1_R1_w]  = w == 0 ? Int32(0) : _find_pos(J, rows, dp, w)

        table.jac_pos[k, J_G1_R2_x1] = _find_pos(J, rows, dp + 1, dp)
        table.jac_pos[k, J_G1_R2_x2] = _find_pos(J, rows, dp + 1, dp + 1)
        table.jac_pos[k, J_G1_R2_w]  = w == 0 ? Int32(0) : _find_pos(J, rows, dp + 1, w)

        table.jac_pos[k, J_G1_R3_x2] = _find_pos(J, rows, dp + 2, dp + 1)
        table.jac_pos[k, J_G1_R3_x3] = _find_pos(J, rows, dp + 2, dp + 2)

        table.jac_pos[k, J_G1_R4_x3] = _find_pos(J, rows, dp + 3, dp + 2)
        table.jac_pos[k, J_G1_R4_x4] = _find_pos(J, rows, dp + 3, dp + 3)

        table.jac_pos[k, J_G1_R5_x4] = _find_pos(J, rows, dp + 4, dp + 3)
        table.jac_pos[k, J_G1_R5_x5] = _find_pos(J, rows, dp + 4, dp + 4)

        table.jac_pos[k, J_G1_R6_x5] = _find_pos(J, rows, dp + 5, dp + 4)
        table.jac_pos[k, J_G1_R6_x6] = _find_pos(J, rows, dp + 5, dp + 5)

        table.jac_pos[k, J_G1_A_x3] = _find_pos(J, rows, ap, dp + 2)
        table.jac_pos[k, J_G1_A_x4] = _find_pos(J, rows, ap, dp + 3)
        table.jac_pos[k, J_G1_A_x5] = _find_pos(J, rows, ap, dp + 4)
        table.jac_pos[k, J_G1_A_x6] = _find_pos(J, rows, ap, dp + 5)
        table.jac_pos[k, J_G1_A_pm] = _find_pos(J, rows, ap, ap)
    end
    return nothing
end

@inline function _ieeeg1_residual_one!(f, z, p,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        k::Int, diff_dim::Int)
    @inbounds begin
    dp = Int(diff_ptr[k])
    ap = Int(alg_ptr[k]) + diff_dim
    pp = Int(par_ptr[k])
    w_idx = Int(w_idx_arr[k])

    K  = p[pp + 2]
    T1 = p[pp + 3]
    T2 = p[pp + 4]
    T3 = p[pp + 5]
    T4 = p[pp + 10]
    K1 = p[pp + 11]
    T5 = p[pp + 13]
    K3 = p[pp + 14]
    T6 = p[pp + 16]
    K5 = p[pp + 17]
    T7 = p[pp + 19]
    K7 = p[pp + 20]
    pref = p[pp + 22]

    x1 = z[dp]
    x2 = z[dp + 1]
    x3 = z[dp + 2]
    x4 = z[dp + 3]
    x5 = z[dp + 4]
    x6 = z[dp + 5]
    p_m = z[ap]
    w = w_idx == 0 ? 0.0 : z[w_idx]

    e   = pref - K * w
    t21 = T2 / T1

    f[dp]     = ((1.0 - t21) * e - x1) / T1
    f[dp + 1] = (x1 + t21 * e - x2) / T3
    f[dp + 2] = (x2 - x3) / T4
    f[dp + 3] = (x3 - x4) / T5
    f[dp + 4] = (x4 - x5) / T6
    f[dp + 5] = (x5 - x6) / T7
    f[ap]     = K1 * x3 + K3 * x4 + K5 * x5 + K7 * x6 - p_m
    end
    return nothing
end

@inline function ieeeg1_residual_batch!(f::AbstractArray, z::AbstractArray,
                                         p::AbstractArray, table::IEEEG1Table,
                                         diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _ieeeg1_residual_one!(f, z, p,
            table.diff_ptr, table.alg_ptr, table.par_ptr, table.w_idx,
            k, diff_dim)
    end
    return nothing
end

@inline function _ieeeg1_jacobian_one!(nz, p, par_ptr, jac_pos, k::Int)
    @inbounds begin
    pp = Int(par_ptr[k])

    K  = p[pp + 2]
    T1 = p[pp + 3]
    T2 = p[pp + 4]
    T3 = p[pp + 5]
    T4 = p[pp + 10]
    K1 = p[pp + 11]
    T5 = p[pp + 13]
    K3 = p[pp + 14]
    T6 = p[pp + 16]
    K5 = p[pp + 17]
    T7 = p[pp + 19]
    K7 = p[pp + 20]

    t21 = T2 / T1

    nz[jac_pos[k, J_G1_R1_x1]] = -1.0 / T1
    pos = jac_pos[k, J_G1_R1_w]
    if pos != 0
        nz[pos] = -K * (1.0 - t21) / T1
    end

    nz[jac_pos[k, J_G1_R2_x1]] =  1.0 / T3
    nz[jac_pos[k, J_G1_R2_x2]] = -1.0 / T3
    pos = jac_pos[k, J_G1_R2_w]
    if pos != 0
        nz[pos] = -K * t21 / T3
    end

    nz[jac_pos[k, J_G1_R3_x2]] =  1.0 / T4
    nz[jac_pos[k, J_G1_R3_x3]] = -1.0 / T4

    nz[jac_pos[k, J_G1_R4_x3]] =  1.0 / T5
    nz[jac_pos[k, J_G1_R4_x4]] = -1.0 / T5

    nz[jac_pos[k, J_G1_R5_x4]] =  1.0 / T6
    nz[jac_pos[k, J_G1_R5_x5]] = -1.0 / T6

    nz[jac_pos[k, J_G1_R6_x5]] =  1.0 / T7
    nz[jac_pos[k, J_G1_R6_x6]] = -1.0 / T7

    nz[jac_pos[k, J_G1_A_x3]] = K1
    nz[jac_pos[k, J_G1_A_x4]] = K3
    nz[jac_pos[k, J_G1_A_x5]] = K5
    nz[jac_pos[k, J_G1_A_x6]] = K7
    nz[jac_pos[k, J_G1_A_pm]] = -1.0
    end
    return nothing
end

@inline function ieeeg1_jacobian_batch!(J::SparseMatrixCSC, p::AbstractArray,
                                         table::IEEEG1Table, diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _ieeeg1_jacobian_one!(nz, p, table.par_ptr, table.jac_pos, k)
    end
    return nothing
end
