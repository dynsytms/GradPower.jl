# HYGOV batched residual & Jacobian.
#
# States: 4 diff + 1 alg (p_m). pref is init-derived (pvec slot 13).
#   dp+0  x1   filtered speed error
#   dp+1  x2   PI integrator
#   dp+2  g    gate opening
#   dp+3  q    penstock flow
#   ap    p_m  mechanical power (algebraic output)
#
# With  c = x1/r + x2,  h = (q/g)^2:
#   F[dp+0] = (pref - w - R*c - x1) / Tf
#   F[dp+1] = x1 / (r*Tr)
#   F[dp+2] = (c - g) / Tg
#   F[dp+3] = (1 - h) / Tw
#   F[ap]   = At*h*(q - qNL) - Dturb*g*w - p_m
#
# NONLINEAR (head/flow), so the Jacobian needs z — same call shape as the
# ESST4B/IEEEST kernels. Divisors arrive already floored at HYGOV_TMIN
# (fill_pvec!), so there is no branch on a zero here. The gate g is > 0
# in any physical operating point (g = pg/At + qNL >= qNL at init).

const HYGOV_JAC_NENTRIES = 13

const J_HY_R1_x1 = 1
const J_HY_R1_x2 = 2
const J_HY_R1_w  = 3
const J_HY_R2_x1 = 4
const J_HY_R3_x1 = 5
const J_HY_R3_x2 = 6
const J_HY_R3_g  = 7
const J_HY_R4_g  = 8
const J_HY_R4_q  = 9
const J_HY_A_g   = 10
const J_HY_A_q   = 11
const J_HY_A_w   = 12
const J_HY_A_pm  = 13

function hygov_preallocate!(coord_list::Vector{Vector{Int}},
                            table::HYGOVTable, diff_dim::Int)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        w  = Int(table.w_idx[k])

        push!(coord_list[dp], dp)
        push!(coord_list[dp], dp + 1)
        w != 0 && push!(coord_list[dp], w)

        push!(coord_list[dp + 1], dp)

        push!(coord_list[dp + 2], dp)
        push!(coord_list[dp + 2], dp + 1)
        push!(coord_list[dp + 2], dp + 2)

        push!(coord_list[dp + 3], dp + 2)
        push!(coord_list[dp + 3], dp + 3)

        push!(coord_list[ap], dp + 2)
        push!(coord_list[ap], dp + 3)
        w != 0 && push!(coord_list[ap], w)
        push!(coord_list[ap], ap)
    end
    return nothing
end

function hygov_jac_positions!(table::HYGOVTable, J::SparseMatrixCSC,
                              diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        w  = Int(table.w_idx[k])

        table.jac_pos[k, J_HY_R1_x1] = _find_pos(J, rows, dp, dp)
        table.jac_pos[k, J_HY_R1_x2] = _find_pos(J, rows, dp, dp + 1)
        table.jac_pos[k, J_HY_R1_w]  = w == 0 ? Int32(0) : _find_pos(J, rows, dp, w)

        table.jac_pos[k, J_HY_R2_x1] = _find_pos(J, rows, dp + 1, dp)

        table.jac_pos[k, J_HY_R3_x1] = _find_pos(J, rows, dp + 2, dp)
        table.jac_pos[k, J_HY_R3_x2] = _find_pos(J, rows, dp + 2, dp + 1)
        table.jac_pos[k, J_HY_R3_g]  = _find_pos(J, rows, dp + 2, dp + 2)

        table.jac_pos[k, J_HY_R4_g]  = _find_pos(J, rows, dp + 3, dp + 2)
        table.jac_pos[k, J_HY_R4_q]  = _find_pos(J, rows, dp + 3, dp + 3)

        table.jac_pos[k, J_HY_A_g]   = _find_pos(J, rows, ap, dp + 2)
        table.jac_pos[k, J_HY_A_q]   = _find_pos(J, rows, ap, dp + 3)
        table.jac_pos[k, J_HY_A_w]   = w == 0 ? Int32(0) : _find_pos(J, rows, ap, w)
        table.jac_pos[k, J_HY_A_pm]  = _find_pos(J, rows, ap, ap)
    end
    return nothing
end

@inline function _hygov_residual_one!(f, z, p,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        k::Int, diff_dim::Int)
    @inbounds begin
    dp = Int(diff_ptr[k])
    ap = Int(alg_ptr[k]) + diff_dim
    pp = Int(par_ptr[k])
    w_idx = Int(w_idx_arr[k])

    R     = p[pp]
    r     = p[pp + 1]
    Tr    = p[pp + 2]
    Tf    = p[pp + 3]
    Tg    = p[pp + 4]
    Tw    = p[pp + 8]
    At    = p[pp + 9]
    Dturb = p[pp + 10]
    qNL   = p[pp + 11]
    pref  = p[pp + 12]

    x1  = z[dp]
    x2  = z[dp + 1]
    g   = z[dp + 2]
    q   = z[dp + 3]
    p_m = z[ap]
    w = w_idx == 0 ? 0.0 : z[w_idx]

    c   = x1 / r + x2
    rho = q / g
    h   = rho * rho

    f[dp]     = (pref - w - R * c - x1) / Tf
    f[dp + 1] = x1 / (r * Tr)
    f[dp + 2] = (c - g) / Tg
    f[dp + 3] = (1.0 - h) / Tw
    f[ap]     = At * h * (q - qNL) - Dturb * g * w - p_m
    end
    return nothing
end

@inline function hygov_residual_batch!(f::AbstractArray, z::AbstractArray,
                                       p::AbstractArray, table::HYGOVTable,
                                       diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _hygov_residual_one!(f, z, p,
            table.diff_ptr, table.alg_ptr, table.par_ptr, table.w_idx,
            k, diff_dim)
    end
    return nothing
end

# Derivatives (rho = q/g, h = rho^2):
#   ∂h/∂q = 2 rho / g,   ∂h/∂g = -2 rho^2 / g = -2 h / g
@inline function _hygov_jacobian_one!(nz, z, p,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr, jac_pos,
        k::Int, diff_dim::Int)
    @inbounds begin
    dp = Int(diff_ptr[k])
    ap = Int(alg_ptr[k]) + diff_dim
    pp = Int(par_ptr[k])
    w_idx = Int(w_idx_arr[k])

    R     = p[pp]
    r     = p[pp + 1]
    Tr    = p[pp + 2]
    Tf    = p[pp + 3]
    Tg    = p[pp + 4]
    Tw    = p[pp + 8]
    At    = p[pp + 9]
    Dturb = p[pp + 10]
    qNL   = p[pp + 11]

    g = z[dp + 2]
    q = z[dp + 3]
    w = w_idx == 0 ? 0.0 : z[w_idx]

    rho   = q / g
    h     = rho * rho
    dh_dq = 2.0 * rho / g
    dh_dg = -2.0 * h / g

    # F1 = (pref - w - R*(x1/r + x2) - x1) / Tf
    nz[jac_pos[k, J_HY_R1_x1]] = -(R / r + 1.0) / Tf
    nz[jac_pos[k, J_HY_R1_x2]] = -R / Tf
    if w_idx != 0
        nz[jac_pos[k, J_HY_R1_w]] = -1.0 / Tf
    end

    # F2 = x1 / (r*Tr)
    nz[jac_pos[k, J_HY_R2_x1]] = 1.0 / (r * Tr)

    # F3 = (x1/r + x2 - g) / Tg
    nz[jac_pos[k, J_HY_R3_x1]] = 1.0 / (r * Tg)
    nz[jac_pos[k, J_HY_R3_x2]] = 1.0 / Tg
    nz[jac_pos[k, J_HY_R3_g]]  = -1.0 / Tg

    # F4 = (1 - h) / Tw
    nz[jac_pos[k, J_HY_R4_g]]  = -dh_dg / Tw
    nz[jac_pos[k, J_HY_R4_q]]  = -dh_dq / Tw

    # Fa = At*h*(q - qNL) - Dturb*g*w - p_m
    nz[jac_pos[k, J_HY_A_g]]   = At * dh_dg * (q - qNL) - Dturb * w
    nz[jac_pos[k, J_HY_A_q]]   = At * (dh_dq * (q - qNL) + h)
    if w_idx != 0
        nz[jac_pos[k, J_HY_A_w]] = -Dturb * g
    end
    nz[jac_pos[k, J_HY_A_pm]]  = -1.0
    end
    return nothing
end

@inline function hygov_jacobian_batch!(J::SparseMatrixCSC, z::AbstractArray,
                                       p::AbstractArray, table::HYGOVTable,
                                       diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _hygov_jacobian_one!(nz, z, p,
            table.diff_ptr, table.alg_ptr, table.par_ptr, table.w_idx,
            table.jac_pos, k, diff_dim)
    end
    return nothing
end
