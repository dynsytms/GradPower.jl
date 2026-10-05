# GGOV1 batched residual & Jacobian.
#
# ⚠️ Governor (PID) path only — load limiter, acceleration limiter and
# the low-value select are omitted. See the scope note above the GGOV1
# struct in src/governors.jl.
#
# States: 5 diff + 1 alg (p_m). pref is init-derived (pvec slot 36).
#   dp+0  x_pelec   electrical-power transducer lag output
#   dp+1  x_igov    governor PI integrator
#   dp+2  x_dgov    governor derivative washout state
#   dp+3  x_act     actuator / valve stroke
#   dp+4  x_tb      turbine lead-lag state
#   ap    p_m       mechanical power (algebraic output)
#
# With  err = pref - w - R*x_pelec,  kd = Kdgov/Tdgov,  tcb = Tc/Tb,
#       fsr = Kpgov*err + x_igov + (kd*err - x_dgov),
#       q   = Kturb*(x_act - Wfnl):
#
#   F[dp+0] = (p_m - x_pelec) / Tpelec
#   F[dp+1] = Kigov * err
#   F[dp+2] = (kd*err - x_dgov) / Tdgov
#   F[dp+3] = (fsr - x_act) / Tact
#   F[dp+4] = ((1 - tcb)*q - x_tb) / Tb
#   F[ap]   = x_tb + tcb*q - Dm*w - p_m
#
# The model is LINEAR in the states, so `_ggov1_jacobian_one!` needs no
# `z` argument — same shape as the TGOV1/IEEEG1 kernels. Time constants
# arrive already floored at GGOV1_TMIN (see `fill_pvec!` and the table
# builder), so there is no branch on T == 0 here.

const GGOV1_JAC_NENTRIES = 18

# row dp+0 (transducer)
const J_GG_R1_xp = 1
const J_GG_R1_pm = 2
# row dp+1 (PI integrator)
const J_GG_R2_xp = 3
const J_GG_R2_w  = 4
# row dp+2 (derivative washout)
const J_GG_R3_xp = 5
const J_GG_R3_xd = 6
const J_GG_R3_w  = 7
# row dp+3 (actuator)
const J_GG_R4_xp = 8
const J_GG_R4_xi = 9
const J_GG_R4_xd = 10
const J_GG_R4_xa = 11
const J_GG_R4_w  = 12
# row dp+4 (turbine lead-lag)
const J_GG_R5_xa = 13
const J_GG_R5_xt = 14
# row ap (algebraic p_m)
const J_GG_A_xa = 15
const J_GG_A_xt = 16
const J_GG_A_pm = 17
const J_GG_A_w  = 18

function ggov1_preallocate!(coord_list::Vector{Vector{Int}},
                            table::GGOV1Table, diff_dim::Int)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        w  = Int(table.w_idx[k])

        # x_pelec' = (p_m - x_pelec)/Tpelec
        push!(coord_list[dp], dp)
        push!(coord_list[dp], ap)

        # x_igov' = Kigov*err
        push!(coord_list[dp + 1], dp)
        w != 0 && push!(coord_list[dp + 1], w)

        # x_dgov' = (kd*err - x_dgov)/Tdgov
        push!(coord_list[dp + 2], dp)
        push!(coord_list[dp + 2], dp + 2)
        w != 0 && push!(coord_list[dp + 2], w)

        # x_act' = (fsr - x_act)/Tact
        push!(coord_list[dp + 3], dp)
        push!(coord_list[dp + 3], dp + 1)
        push!(coord_list[dp + 3], dp + 2)
        push!(coord_list[dp + 3], dp + 3)
        w != 0 && push!(coord_list[dp + 3], w)

        # x_tb' = ((1-tcb)*q - x_tb)/Tb
        push!(coord_list[dp + 4], dp + 3)
        push!(coord_list[dp + 4], dp + 4)

        # p_m algebraic row
        push!(coord_list[ap], dp + 3)
        push!(coord_list[ap], dp + 4)
        push!(coord_list[ap], ap)
        w != 0 && push!(coord_list[ap], w)
    end
    return nothing
end

function ggov1_jac_positions!(table::GGOV1Table, J::SparseMatrixCSC,
                              diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        w  = Int(table.w_idx[k])

        table.jac_pos[k, J_GG_R1_xp] = _find_pos(J, rows, dp, dp)
        table.jac_pos[k, J_GG_R1_pm] = _find_pos(J, rows, dp, ap)

        table.jac_pos[k, J_GG_R2_xp] = _find_pos(J, rows, dp + 1, dp)
        table.jac_pos[k, J_GG_R2_w]  = w == 0 ? Int32(0) : _find_pos(J, rows, dp + 1, w)

        table.jac_pos[k, J_GG_R3_xp] = _find_pos(J, rows, dp + 2, dp)
        table.jac_pos[k, J_GG_R3_xd] = _find_pos(J, rows, dp + 2, dp + 2)
        table.jac_pos[k, J_GG_R3_w]  = w == 0 ? Int32(0) : _find_pos(J, rows, dp + 2, w)

        table.jac_pos[k, J_GG_R4_xp] = _find_pos(J, rows, dp + 3, dp)
        table.jac_pos[k, J_GG_R4_xi] = _find_pos(J, rows, dp + 3, dp + 1)
        table.jac_pos[k, J_GG_R4_xd] = _find_pos(J, rows, dp + 3, dp + 2)
        table.jac_pos[k, J_GG_R4_xa] = _find_pos(J, rows, dp + 3, dp + 3)
        table.jac_pos[k, J_GG_R4_w]  = w == 0 ? Int32(0) : _find_pos(J, rows, dp + 3, w)

        table.jac_pos[k, J_GG_R5_xa] = _find_pos(J, rows, dp + 4, dp + 3)
        table.jac_pos[k, J_GG_R5_xt] = _find_pos(J, rows, dp + 4, dp + 4)

        table.jac_pos[k, J_GG_A_xa] = _find_pos(J, rows, ap, dp + 3)
        table.jac_pos[k, J_GG_A_xt] = _find_pos(J, rows, ap, dp + 4)
        table.jac_pos[k, J_GG_A_pm] = _find_pos(J, rows, ap, ap)
        table.jac_pos[k, J_GG_A_w]  = w == 0 ? Int32(0) : _find_pos(J, rows, ap, w)
    end
    return nothing
end

@inline function _ggov1_residual_one!(f, z, p,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        k::Int, diff_dim::Int)
    @inbounds begin
    dp = Int(diff_ptr[k])
    ap = Int(alg_ptr[k]) + diff_dim
    pp = Int(par_ptr[k])
    w_idx = Int(w_idx_arr[k])

    R      = p[pp + 2]
    Tpelec = p[pp + 3]
    Kpgov  = p[pp + 6]
    Kigov  = p[pp + 7]
    Kdgov  = p[pp + 8]
    Tdgov  = p[pp + 9]
    Tact   = p[pp + 12]
    Kturb  = p[pp + 13]
    Wfnl   = p[pp + 14]
    Tb     = p[pp + 15]
    Tc     = p[pp + 16]
    Dm     = p[pp + 22]
    pref   = p[pp + 35]

    x_pelec = z[dp]
    x_igov  = z[dp + 1]
    x_dgov  = z[dp + 2]
    x_act   = z[dp + 3]
    x_tb    = z[dp + 4]
    p_m     = z[ap]
    w = w_idx == 0 ? 0.0 : z[w_idx]

    err  = pref - w - R * x_pelec
    kd   = Kdgov / Tdgov
    dgov = kd * err - x_dgov
    fsr  = Kpgov * err + x_igov + dgov
    q    = Kturb * (x_act - Wfnl)
    tcb  = Tc / Tb

    f[dp]     = (p_m - x_pelec) / Tpelec
    f[dp + 1] = Kigov * err
    f[dp + 2] = dgov / Tdgov
    f[dp + 3] = (fsr - x_act) / Tact
    f[dp + 4] = ((1.0 - tcb) * q - x_tb) / Tb
    f[ap]     = x_tb + tcb * q - Dm * w - p_m
    end
    return nothing
end

@inline function ggov1_residual_batch!(f::AbstractArray, z::AbstractArray,
                                       p::AbstractArray, table::GGOV1Table,
                                       diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _ggov1_residual_one!(f, z, p,
            table.diff_ptr, table.alg_ptr, table.par_ptr, table.w_idx,
            k, diff_dim)
    end
    return nothing
end

@inline function _ggov1_jacobian_one!(nz, p, par_ptr, jac_pos, k::Int)
    @inbounds begin
    pp = Int(par_ptr[k])

    R      = p[pp + 2]
    Tpelec = p[pp + 3]
    Kpgov  = p[pp + 6]
    Kigov  = p[pp + 7]
    Kdgov  = p[pp + 8]
    Tdgov  = p[pp + 9]
    Tact   = p[pp + 12]
    Kturb  = p[pp + 13]
    Tb     = p[pp + 15]
    Tc     = p[pp + 16]
    Dm     = p[pp + 22]

    kd  = Kdgov / Tdgov      # derivative feed-forward gain
    kpd = Kpgov + kd         # total proportional gain into the actuator
    tcb = Tc / Tb

    # d err / d x_pelec = -R ; d err / d w = -1

    nz[jac_pos[k, J_GG_R1_xp]] = -1.0 / Tpelec
    nz[jac_pos[k, J_GG_R1_pm]] =  1.0 / Tpelec

    nz[jac_pos[k, J_GG_R2_xp]] = -Kigov * R
    pos = jac_pos[k, J_GG_R2_w]
    if pos != 0
        nz[pos] = -Kigov
    end

    nz[jac_pos[k, J_GG_R3_xp]] = -kd * R / Tdgov
    nz[jac_pos[k, J_GG_R3_xd]] = -1.0 / Tdgov
    pos = jac_pos[k, J_GG_R3_w]
    if pos != 0
        nz[pos] = -kd / Tdgov
    end

    nz[jac_pos[k, J_GG_R4_xp]] = -kpd * R / Tact
    nz[jac_pos[k, J_GG_R4_xi]] =  1.0 / Tact
    nz[jac_pos[k, J_GG_R4_xd]] = -1.0 / Tact
    nz[jac_pos[k, J_GG_R4_xa]] = -1.0 / Tact
    pos = jac_pos[k, J_GG_R4_w]
    if pos != 0
        nz[pos] = -kpd / Tact
    end

    nz[jac_pos[k, J_GG_R5_xa]] = (1.0 - tcb) * Kturb / Tb
    nz[jac_pos[k, J_GG_R5_xt]] = -1.0 / Tb

    nz[jac_pos[k, J_GG_A_xa]] = tcb * Kturb
    nz[jac_pos[k, J_GG_A_xt]] = 1.0
    nz[jac_pos[k, J_GG_A_pm]] = -1.0
    pos = jac_pos[k, J_GG_A_w]
    if pos != 0
        nz[pos] = -Dm
    end
    end
    return nothing
end

@inline function ggov1_jacobian_batch!(J::SparseMatrixCSC, p::AbstractArray,
                                       table::GGOV1Table, diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _ggov1_jacobian_one!(nz, p, table.par_ptr, table.jac_pos, k)
    end
    return nothing
end
