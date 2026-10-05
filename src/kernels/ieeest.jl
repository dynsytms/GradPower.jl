# IEEEST batched residual & Jacobian.
#
# States: 7 diff (s0..s6) + 1 alg (v_s). Reads the speed deviation w from
# z[w_idx]. See src/stabilizers.jl for the realization and the OUTPUT LIMIT
# note (the smooth LSMAX/LSMIN saturation is applied here).
#
#   s0, s1 : F1 = N/D1 = (1 + A5 s + A6 s^2)/(1 + A1 s + A2 s^2)
#   s2, s3 : F2 = 1/D2 = 1/(1 + A3 s + A4 s^2)
#   s4     : LL1 (T1/T2),   s5 : LL2 (T3/T4),   s6 : washout (T5/T6, KS)
#   vs     : v_s (alg, at diff_dim + alg_ptr)
#
# Inline intermediates (not z-vector states):
#   q1 = sig - s1 - A1*s0
#   y1 = s1 + A5*s0 + (A6/A2)*q1
#   y2 = s3
#   y3 = s4 + (T1/T2)*(y2 - s4)
#   y4 = s5 + (T3/T4)*(y3 - s5)
#   x  = (T5/T6)*(KS*y4 - s6)
#
# Residual equations (sig = w):
#   f[dp+0] = q1 / A2
#   f[dp+1] = s0
#   f[dp+2] = (y1 - s3 - A3*s2) / A4
#   f[dp+3] = s2
#   f[dp+4] = (y2 - s4) / T2
#   f[dp+5] = (y3 - s5) / T4
#   f[dp+6] = (KS*y4 - s6) / T6
#   f[ap]   = vs - sat(x)
#
# sat(x) = c + h*tanh(x/h - atanh(c/h)), c/h from LSMAX/LSMIN (p[pp+13],
# p[pp+14]; degenerate limits already replaced by +/-1e3 in fill_pvec!).
#
# Jacobian entries per device (24 total):
#   f0:   ∂/∂{s0, s1, w}                     3
#   f1:   ∂/∂{s0}                            1
#   f2:   ∂/∂{s0, s1, s2, s3, w}             5
#   f3:   ∂/∂{s2}                            1
#   f4:   ∂/∂{s3, s4}                        2
#   f5:   ∂/∂{s3, s4, s5}                    3
#   f6:   ∂/∂{s3, s4, s5, s6}                4
#   f_vs: ∂/∂{s3, s4, s5, s6, vs}            5   (scaled by sat'(x))
#                                     Total: 24

const IEEEST_JAC_NENTRIES = 24

# Slot indices into jac_pos (1-based).
const J_PSS_R0_s0    = 1
const J_PSS_R0_s1    = 2
const J_PSS_R0_w     = 3
const J_PSS_R1_s0    = 4
const J_PSS_R2_s0    = 5
const J_PSS_R2_s1    = 6
const J_PSS_R2_s2    = 7
const J_PSS_R2_s3    = 8
const J_PSS_R2_w     = 9
const J_PSS_R3_s2    = 10
const J_PSS_R4_s3    = 11
const J_PSS_R4_s4    = 12
const J_PSS_R5_s3    = 13
const J_PSS_R5_s4    = 14
const J_PSS_R5_s5    = 15
const J_PSS_R6_s3    = 16
const J_PSS_R6_s4    = 17
const J_PSS_R6_s5    = 18
const J_PSS_R6_s6    = 19
const J_PSS_VA_s3    = 20
const J_PSS_VA_s4    = 21
const J_PSS_VA_s5    = 22
const J_PSS_VA_s6    = 23
const J_PSS_VA_vs    = 24

# (row offset, col) pattern shared by preallocate and position cache.
# Row offsets 0..6 are diff rows dp+r; row offset -1 denotes the alg row.
# Column codes 0..6 are s0..s6, -1 = w (omega), -2 = vs.
const _IEEEST_PATTERN = (
    (0, 0), (0, 1), (0, -1),
    (1, 0),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, -1),
    (3, 2),
    (4, 3), (4, 4),
    (5, 3), (5, 4), (5, 5),
    (6, 3), (6, 4), (6, 5), (6, 6),
    (-1, 3), (-1, 4), (-1, 5), (-1, 6), (-1, -2),
)
@assert length(_IEEEST_PATTERN) == IEEEST_JAC_NENTRIES

@inline _ieeest_row(r, dp, ap) = r < 0 ? ap : dp + r
@inline _ieeest_col(c, dp, ap, wi) = c == -1 ? wi : (c == -2 ? ap : dp + c)

# --------------------------------------------------------------------
# Sparsity contribution
# --------------------------------------------------------------------

function ieeest_preallocate!(coord_list::Vector{Vector{Int}},
                              table::IEEESTTable, diff_dim::Int)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        ap = diff_dim + Int(table.alg_ptr[k])
        wi = Int(table.w_idx[k])
        for (r, c) in _IEEEST_PATTERN
            col = _ieeest_col(c, dp, ap, wi)
            col > 0 || continue          # unwired omega
            push!(coord_list[_ieeest_row(r, dp, ap)], col)
        end
    end
    return nothing
end

# --------------------------------------------------------------------
# Position cache
# --------------------------------------------------------------------

function ieeest_jac_positions!(table::IEEESTTable, J::SparseMatrixCSC, diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        ap = diff_dim + Int(table.alg_ptr[k])
        wi = Int(table.w_idx[k])
        for (slot, (r, c)) in enumerate(_IEEEST_PATTERN)
            col = _ieeest_col(c, dp, ap, wi)
            table.jac_pos[k, slot] = col > 0 ?
                _find_pos(J, rows, _ieeest_row(r, dp, ap), col) : Int32(0)
        end
    end
    return nothing
end

# --------------------------------------------------------------------
# Inline helper: chain of algebraic intermediates
# --------------------------------------------------------------------

@inline function _ieeest_chain(sig, s0, s1, s3, s4, s5,
                                A1, A2, A5, A6, T1, T2, T3, T4)
    q1 = sig - s1 - A1*s0
    y1 = s1 + A5*s0 + (A6/A2)*q1
    y2 = s3
    y3 = s4 + (T1/T2)*(y2 - s4)
    y4 = s5 + (T3/T4)*(y3 - s5)
    return q1, y1, y3, y4
end

# --------------------------------------------------------------------
# Residual batch
# --------------------------------------------------------------------

@inline function _ieeest_residual_one!(f, z, p,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        diff_dim, k::Int)
    @inbounds begin
    dp = Int(diff_ptr[k])
    ap = diff_dim + Int(alg_ptr[k])
    pp = Int(par_ptr[k])
    wi = Int(w_idx_arr[k])

    A1 = p[pp];     A2 = p[pp+1];  A3 = p[pp+2];  A4 = p[pp+3]
    A5 = p[pp+4];   A6 = p[pp+5]
    T1 = p[pp+6];   T2 = p[pp+7];  T3 = p[pp+8];  T4 = p[pp+9]
    T5 = p[pp+10];  T6 = p[pp+11]; KS = p[pp+12]
    LSMAX = p[pp+13]; LSMIN = p[pp+14]

    s0 = z[dp];   s1 = z[dp+1]; s2 = z[dp+2]; s3 = z[dp+3]
    s4 = z[dp+4]; s5 = z[dp+5]; s6 = z[dp+6]
    vs = z[ap]

    # w_idx points to Genrou's w state, which is the speed DEVIATION
    # (omega - 1). So sig = w directly (no subtraction needed).
    sig = wi > 0 ? z[wi] : 0.0

    q1, y1, y3, y4 = _ieeest_chain(sig, s0, s1, s3, s4, s5,
                                   A1, A2, A5, A6, T1, T2, T3, T4)
    vsat, _ = _ieeest_sat((T5/T6)*(KS*y4 - s6), LSMAX, LSMIN)

    f[dp]   = q1 / A2
    f[dp+1] = s0
    f[dp+2] = (y1 - s3 - A3*s2) / A4
    f[dp+3] = s2
    f[dp+4] = (s3 - s4) / T2
    f[dp+5] = (y3 - s5) / T4
    f[dp+6] = (KS*y4 - s6) / T6
    f[ap]   = vs - vsat
    end
    return nothing
end

@inline function ieeest_residual_batch!(f::AbstractArray, z::AbstractArray,
                                         p::AbstractArray, table::IEEESTTable,
                                         diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _ieeest_residual_one!(f, z, p,
            table.diff_ptr, table.alg_ptr, table.par_ptr, table.w_idx,
            diff_dim, k)
    end
    return nothing
end

# --------------------------------------------------------------------
# Jacobian batch
# --------------------------------------------------------------------

# Derivatives of the inline intermediates:
#   y1: ∂/∂s0 = A5 - A6*A1/A2,  ∂/∂s1 = 1 - A6/A2,  ∂/∂w = A6/A2
#   y3: ∂/∂s3 = r12,  ∂/∂s4 = 1 - r12                    (r12 = T1/T2)
#   y4: ∂/∂s3 = r34*r12,  ∂/∂s4 = r34*(1 - r12),  ∂/∂s5 = 1 - r34
#   x = r56*(KS*y4 - s6),  v_s = sat(x),  sat'(x) = 1 - tanh^2

@inline function _ieeest_jacobian_one!(nz, z, p,
        par_ptr, diff_ptr, alg_ptr, w_idx_arr, jac_pos,
        diff_dim, k::Int)
    @inbounds begin
    pp = Int(par_ptr[k])
    dp = Int(diff_ptr[k])
    wi = Int(w_idx_arr[k])

    A1 = p[pp];   A2 = p[pp+1]; A3 = p[pp+2]; A4 = p[pp+3]
    A5 = p[pp+4]; A6 = p[pp+5]
    T1 = p[pp+6]; T2 = p[pp+7]; T3 = p[pp+8]; T4 = p[pp+9]
    T5 = p[pp+10]; T6 = p[pp+11]; KS = p[pp+12]
    LSMAX = p[pp+13]; LSMIN = p[pp+14]

    s0 = z[dp]; s1 = z[dp+1]; s3 = z[dp+3]
    s4 = z[dp+4]; s5 = z[dp+5]; s6 = z[dp+6]
    sig = wi > 0 ? z[wi] : 0.0

    r6 = A6 / A2
    dy1_ds0 = A5 - r6*A1
    dy1_ds1 = 1.0 - r6
    dy1_dw  = r6

    r12 = T1 / T2
    dy3_ds3 = r12
    dy3_ds4 = 1.0 - r12

    r34 = T3 / T4
    dy4_ds3 = r34 * dy3_ds3
    dy4_ds4 = r34 * dy3_ds4
    dy4_ds5 = 1.0 - r34

    # Row dp+0: f0 = (w - s1 - A1*s0) / A2
    nz[jac_pos[k, J_PSS_R0_s0]] = -A1 / A2
    nz[jac_pos[k, J_PSS_R0_s1]] = -1.0 / A2
    if wi > 0
        nz[jac_pos[k, J_PSS_R0_w]] = 1.0 / A2
    end

    # Row dp+1: f1 = s0
    nz[jac_pos[k, J_PSS_R1_s0]] = 1.0

    # Row dp+2: f2 = (y1 - s3 - A3*s2) / A4
    nz[jac_pos[k, J_PSS_R2_s0]] = dy1_ds0 / A4
    nz[jac_pos[k, J_PSS_R2_s1]] = dy1_ds1 / A4
    nz[jac_pos[k, J_PSS_R2_s2]] = -A3 / A4
    nz[jac_pos[k, J_PSS_R2_s3]] = -1.0 / A4
    if wi > 0
        nz[jac_pos[k, J_PSS_R2_w]] = dy1_dw / A4
    end

    # Row dp+3: f3 = s2
    nz[jac_pos[k, J_PSS_R3_s2]] = 1.0

    # Row dp+4: f4 = (s3 - s4) / T2
    nz[jac_pos[k, J_PSS_R4_s3]] = 1.0 / T2
    nz[jac_pos[k, J_PSS_R4_s4]] = -1.0 / T2

    # Row dp+5: f5 = (y3 - s5) / T4
    nz[jac_pos[k, J_PSS_R5_s3]] = dy3_ds3 / T4
    nz[jac_pos[k, J_PSS_R5_s4]] = dy3_ds4 / T4
    nz[jac_pos[k, J_PSS_R5_s5]] = -1.0 / T4

    # Row dp+6: f6 = (KS*y4 - s6) / T6
    nz[jac_pos[k, J_PSS_R6_s3]] = KS * dy4_ds3 / T6
    nz[jac_pos[k, J_PSS_R6_s4]] = KS * dy4_ds4 / T6
    nz[jac_pos[k, J_PSS_R6_s5]] = KS * dy4_ds5 / T6
    nz[jac_pos[k, J_PSS_R6_s6]] = -1.0 / T6

    # Alg row: f_vs = vs - sat(x),  x = (T5/T6)*(KS*y4 - s6)
    r56 = T5 / T6
    _, _, _, y4 = _ieeest_chain(sig, s0, s1, s3, s4, s5,
                                A1, A2, A5, A6, T1, T2, T3, T4)
    _, dsat = _ieeest_sat(r56*(KS*y4 - s6), LSMAX, LSMIN)
    g = r56 * dsat
    nz[jac_pos[k, J_PSS_VA_s3]] = -g * KS * dy4_ds3
    nz[jac_pos[k, J_PSS_VA_s4]] = -g * KS * dy4_ds4
    nz[jac_pos[k, J_PSS_VA_s5]] = -g * KS * dy4_ds5
    nz[jac_pos[k, J_PSS_VA_s6]] = g
    nz[jac_pos[k, J_PSS_VA_vs]] = 1.0
    end
    return nothing
end

@inline function ieeest_jacobian_batch!(J::SparseMatrixCSC, z::AbstractArray,
                                         p::AbstractArray, table::IEEESTTable,
                                         diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _ieeest_jacobian_one!(nz, z, p,
            table.par_ptr, table.diff_ptr, table.alg_ptr, table.w_idx, table.jac_pos,
            diff_dim, k)
    end
    return nothing
end
