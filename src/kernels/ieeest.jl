# IEEEST batched residual & Jacobian.
#
# States: 7 diff (s0..s6) + 1 alg (v_s). Reads omega from z[w_idx].
#
# State names:
#   s0 = F1_x   (Lag2ndOrd internal)
#   s1 = F1_y   (Lag2ndOrd output)
#   s2 = F2_x1  (LeadLag2ndOrd state 1)
#   s3 = F2_x2  (LeadLag2ndOrd state 2)
#   s4 = LL1_x  (LeadLag 1 state)
#   s5 = LL2_x  (LeadLag 2 state)
#   s6 = WO_x   (Washout state)
#   vs = v_s    (alg: PSS output, at diff_dim + alg_ptr)
#
# Inline algebraic intermediates (not z-vector states):
#   y2 = s3 + A5*s2 + (A6/A4)*(s1 - s3 - A3*s2)
#   y3 = s4 + (T1/T2)*(y2 - s4)
#   y4 = s5 + (T3/T4)*(y3 - s5)
#
# Residual equations (sig = omega - 1):
#   f[dp+0] = (sig - s1 - A1*s0) / A2
#   f[dp+1] = s0
#   f[dp+2] = (s1 - s3 - A3*s2) / A4
#   f[dp+3] = s2
#   f[dp+4] = (y2 - s4) / T2
#   f[dp+5] = (y3 - s5) / T4
#   f[dp+6] = (KS*y4 - s6) / T6
#   f[ap]   = vs - (T5/T6)*(KS*y4 - s6)
#
# Jacobian entries per device (30 total):
#   f0: ∂/∂{s0, s1, omega}                            3
#   f1: ∂/∂{s0}                                       1
#   f2: ∂/∂{s1, s2, s3}                               3
#   f3: ∂/∂{s2}                                       1
#   f4: ∂/∂{s1, s2, s3, s4}                           4
#   f5: ∂/∂{s1, s2, s3, s4, s5}                       5
#   f6: ∂/∂{s1, s2, s3, s4, s5, s6}                   6
#   f_vs: ∂/∂{s1, s2, s3, s4, s5, s6, vs}             7
#                                              Total: 30

const IEEEST_JAC_NENTRIES = 30

# Slot indices into jac_pos (1-based).
const J_PSS_R0_s0    = 1
const J_PSS_R0_s1    = 2
const J_PSS_R0_omega = 3
const J_PSS_R1_s0    = 4
const J_PSS_R2_s1    = 5
const J_PSS_R2_s2    = 6
const J_PSS_R2_s3    = 7
const J_PSS_R3_s2    = 8
const J_PSS_R4_s1    = 9
const J_PSS_R4_s2    = 10
const J_PSS_R4_s3    = 11
const J_PSS_R4_s4    = 12
const J_PSS_R5_s1    = 13
const J_PSS_R5_s2    = 14
const J_PSS_R5_s3    = 15
const J_PSS_R5_s4    = 16
const J_PSS_R5_s5    = 17
const J_PSS_R6_s1    = 18
const J_PSS_R6_s2    = 19
const J_PSS_R6_s3    = 20
const J_PSS_R6_s4    = 21
const J_PSS_R6_s5    = 22
const J_PSS_R6_s6    = 23
const J_PSS_VA_s1    = 24
const J_PSS_VA_s2    = 25
const J_PSS_VA_s3    = 26
const J_PSS_VA_s4    = 27
const J_PSS_VA_s5    = 28
const J_PSS_VA_s6    = 29
const J_PSS_VA_vs    = 30

# --------------------------------------------------------------------
# Sparsity contribution
# --------------------------------------------------------------------

function ieeest_preallocate!(coord_list::Vector{Vector{Int}},
                              table::IEEESTTable, diff_dim::Int)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        ap = diff_dim + Int(table.alg_ptr[k])
        wi = Int(table.w_idx[k])

        # f0: s0, s1, omega
        push!(coord_list[dp],     dp)
        push!(coord_list[dp],     dp + 1)
        wi > 0 && push!(coord_list[dp], wi)

        # f1: s0
        push!(coord_list[dp + 1], dp)

        # f2: s1, s2, s3
        push!(coord_list[dp + 2], dp + 1)
        push!(coord_list[dp + 2], dp + 2)
        push!(coord_list[dp + 2], dp + 3)

        # f3: s2
        push!(coord_list[dp + 3], dp + 2)

        # f4: s1, s2, s3, s4
        push!(coord_list[dp + 4], dp + 1)
        push!(coord_list[dp + 4], dp + 2)
        push!(coord_list[dp + 4], dp + 3)
        push!(coord_list[dp + 4], dp + 4)

        # f5: s1, s2, s3, s4, s5
        push!(coord_list[dp + 5], dp + 1)
        push!(coord_list[dp + 5], dp + 2)
        push!(coord_list[dp + 5], dp + 3)
        push!(coord_list[dp + 5], dp + 4)
        push!(coord_list[dp + 5], dp + 5)

        # f6: s1, s2, s3, s4, s5, s6
        push!(coord_list[dp + 6], dp + 1)
        push!(coord_list[dp + 6], dp + 2)
        push!(coord_list[dp + 6], dp + 3)
        push!(coord_list[dp + 6], dp + 4)
        push!(coord_list[dp + 6], dp + 5)
        push!(coord_list[dp + 6], dp + 6)

        # f_vs (alg row): s1, s2, s3, s4, s5, s6, vs
        push!(coord_list[ap], dp + 1)
        push!(coord_list[ap], dp + 2)
        push!(coord_list[ap], dp + 3)
        push!(coord_list[ap], dp + 4)
        push!(coord_list[ap], dp + 5)
        push!(coord_list[ap], dp + 6)
        push!(coord_list[ap], ap)
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

        table.jac_pos[k, J_PSS_R0_s0]    = _find_pos(J, rows, dp,     dp)
        table.jac_pos[k, J_PSS_R0_s1]    = _find_pos(J, rows, dp,     dp + 1)
        table.jac_pos[k, J_PSS_R0_omega] = wi > 0 ? _find_pos(J, rows, dp, wi) : Int32(0)
        table.jac_pos[k, J_PSS_R1_s0]    = _find_pos(J, rows, dp + 1, dp)

        table.jac_pos[k, J_PSS_R2_s1]    = _find_pos(J, rows, dp + 2, dp + 1)
        table.jac_pos[k, J_PSS_R2_s2]    = _find_pos(J, rows, dp + 2, dp + 2)
        table.jac_pos[k, J_PSS_R2_s3]    = _find_pos(J, rows, dp + 2, dp + 3)

        table.jac_pos[k, J_PSS_R3_s2]    = _find_pos(J, rows, dp + 3, dp + 2)

        table.jac_pos[k, J_PSS_R4_s1]    = _find_pos(J, rows, dp + 4, dp + 1)
        table.jac_pos[k, J_PSS_R4_s2]    = _find_pos(J, rows, dp + 4, dp + 2)
        table.jac_pos[k, J_PSS_R4_s3]    = _find_pos(J, rows, dp + 4, dp + 3)
        table.jac_pos[k, J_PSS_R4_s4]    = _find_pos(J, rows, dp + 4, dp + 4)

        table.jac_pos[k, J_PSS_R5_s1]    = _find_pos(J, rows, dp + 5, dp + 1)
        table.jac_pos[k, J_PSS_R5_s2]    = _find_pos(J, rows, dp + 5, dp + 2)
        table.jac_pos[k, J_PSS_R5_s3]    = _find_pos(J, rows, dp + 5, dp + 3)
        table.jac_pos[k, J_PSS_R5_s4]    = _find_pos(J, rows, dp + 5, dp + 4)
        table.jac_pos[k, J_PSS_R5_s5]    = _find_pos(J, rows, dp + 5, dp + 5)

        table.jac_pos[k, J_PSS_R6_s1]    = _find_pos(J, rows, dp + 6, dp + 1)
        table.jac_pos[k, J_PSS_R6_s2]    = _find_pos(J, rows, dp + 6, dp + 2)
        table.jac_pos[k, J_PSS_R6_s3]    = _find_pos(J, rows, dp + 6, dp + 3)
        table.jac_pos[k, J_PSS_R6_s4]    = _find_pos(J, rows, dp + 6, dp + 4)
        table.jac_pos[k, J_PSS_R6_s5]    = _find_pos(J, rows, dp + 6, dp + 5)
        table.jac_pos[k, J_PSS_R6_s6]    = _find_pos(J, rows, dp + 6, dp + 6)

        table.jac_pos[k, J_PSS_VA_s1]    = _find_pos(J, rows, ap, dp + 1)
        table.jac_pos[k, J_PSS_VA_s2]    = _find_pos(J, rows, ap, dp + 2)
        table.jac_pos[k, J_PSS_VA_s3]    = _find_pos(J, rows, ap, dp + 3)
        table.jac_pos[k, J_PSS_VA_s4]    = _find_pos(J, rows, ap, dp + 4)
        table.jac_pos[k, J_PSS_VA_s5]    = _find_pos(J, rows, ap, dp + 5)
        table.jac_pos[k, J_PSS_VA_s6]    = _find_pos(J, rows, ap, dp + 6)
        table.jac_pos[k, J_PSS_VA_vs]    = _find_pos(J, rows, ap, ap)
    end
    return nothing
end

# --------------------------------------------------------------------
# Inline helpers: compute chain of algebraic intermediates
# --------------------------------------------------------------------

# Compute y2, y3, y4 from diff states and parameters.
@inline function _ieeest_chain(s1, s2, s3, s4, s5,
                                A3, A4, A5, A6,
                                T1, T2, T3, T4)
    q = s1 - s3 - A3*s2           # reused: F2 ODE RHS numerator
    y2 = s3 + A5*s2 + (A6/A4)*q
    y3 = s4 + (T1/T2)*(y2 - s4)
    y4 = s5 + (T3/T4)*(y3 - s5)
    return y2, y3, y4
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

    s0 = z[dp];   s1 = z[dp+1]; s2 = z[dp+2]; s3 = z[dp+3]
    s4 = z[dp+4]; s5 = z[dp+5]; s6 = z[dp+6]
    vs = z[ap]

    # w_idx points to Genrou's w state, which is the speed DEVIATION
    # (omega - 1). So sig = w directly (no subtraction needed).
    sig = wi > 0 ? z[wi] : 0.0

    y2, y3, y4 = _ieeest_chain(s1, s2, s3, s4, s5, A3, A4, A5, A6, T1, T2, T3, T4)

    f[dp]   = (sig - s1 - A1*s0) / A2
    f[dp+1] = s0
    f[dp+2] = (s1 - s3 - A3*s2) / A4
    f[dp+3] = s2
    f[dp+4] = (y2 - s4) / T2
    f[dp+5] = (y3 - s5) / T4
    f[dp+6] = (KS*y4 - s6) / T6
    f[ap]   = vs - (T5/T6)*(KS*y4 - s6)
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

# Derivative chain for the inline algebraic intermediates.
#
# y2 = s3 + A5*s2 + (A6/A4)*(s1 - s3 - A3*s2)
#   ∂y2/∂s1 = A6/A4
#   ∂y2/∂s2 = A5 - A3*A6/A4
#   ∂y2/∂s3 = 1 - A6/A4
#
# y3 = s4 + (T1/T2)*(y2 - s4)  = (1 - T1/T2)*s4 + (T1/T2)*y2
#   ∂y3/∂s4 = 1 - T1/T2
#   ∂y3/∂s_j = (T1/T2)*∂y2/∂s_j    for j ∈ {1,2,3}
#
# y4 = s5 + (T3/T4)*(y3 - s5) = (1 - T3/T4)*s5 + (T3/T4)*y3
#   ∂y4/∂s5 = 1 - T3/T4
#   ∂y4/∂s_j = (T3/T4)*∂y3/∂s_j    for j ∈ {1,2,3,4}

@inline function _ieeest_jacobian_one!(nz, z, p,
        par_ptr, diff_ptr, alg_ptr, w_idx_arr, jac_pos,
        diff_dim, k::Int)
    @inbounds begin
    pp = Int(par_ptr[k])
    dp = Int(diff_ptr[k])
    ap = diff_dim + Int(alg_ptr[k])
    wi = Int(w_idx_arr[k])

    A1 = p[pp];   A2 = p[pp+1]; A3 = p[pp+2]; A4 = p[pp+3]
    A5 = p[pp+4]; A6 = p[pp+5]
    T1 = p[pp+6]; T2 = p[pp+7]; T3 = p[pp+8]; T4 = p[pp+9]
    T5 = p[pp+10]; T6 = p[pp+11]; KS = p[pp+12]

    # Derivative chain for inline algebraic intermediates
    rA = A6/A4
    dy2_ds1 = rA
    dy2_ds2 = A5 - A3*rA
    dy2_ds3 = 1.0 - rA

    rT12 = T1/T2
    dy3_ds1 = rT12 * dy2_ds1
    dy3_ds2 = rT12 * dy2_ds2
    dy3_ds3 = rT12 * dy2_ds3
    dy3_ds4 = 1.0 - rT12

    rT34 = T3/T4
    dy4_ds1 = rT34 * dy3_ds1
    dy4_ds2 = rT34 * dy3_ds2
    dy4_ds3 = rT34 * dy3_ds3
    dy4_ds4 = rT34 * dy3_ds4
    dy4_ds5 = 1.0 - rT34

    # Row dp+0: f0 = (sig - s1 - A1*s0) / A2
    nz[jac_pos[k, J_PSS_R0_s0]] = -A1 / A2
    nz[jac_pos[k, J_PSS_R0_s1]] = -1.0 / A2
    if wi > 0
        nz[jac_pos[k, J_PSS_R0_omega]] = 1.0 / A2
    end

    # Row dp+1: f1 = s0
    nz[jac_pos[k, J_PSS_R1_s0]] = 1.0

    # Row dp+2: f2 = (s1 - s3 - A3*s2) / A4
    nz[jac_pos[k, J_PSS_R2_s1]] = 1.0 / A4
    nz[jac_pos[k, J_PSS_R2_s2]] = -A3 / A4
    nz[jac_pos[k, J_PSS_R2_s3]] = -1.0 / A4

    # Row dp+3: f3 = s2
    nz[jac_pos[k, J_PSS_R3_s2]] = 1.0

    # Row dp+4: f4 = (y2 - s4) / T2
    nz[jac_pos[k, J_PSS_R4_s1]] = dy2_ds1 / T2
    nz[jac_pos[k, J_PSS_R4_s2]] = dy2_ds2 / T2
    nz[jac_pos[k, J_PSS_R4_s3]] = dy2_ds3 / T2
    nz[jac_pos[k, J_PSS_R4_s4]] = -1.0 / T2

    # Row dp+5: f5 = (y3 - s5) / T4
    nz[jac_pos[k, J_PSS_R5_s1]] = dy3_ds1 / T4
    nz[jac_pos[k, J_PSS_R5_s2]] = dy3_ds2 / T4
    nz[jac_pos[k, J_PSS_R5_s3]] = dy3_ds3 / T4
    nz[jac_pos[k, J_PSS_R5_s4]] = dy3_ds4 / T4
    nz[jac_pos[k, J_PSS_R5_s5]] = -1.0 / T4

    # Row dp+6: f6 = (KS*y4 - s6) / T6
    nz[jac_pos[k, J_PSS_R6_s1]] = KS * dy4_ds1 / T6
    nz[jac_pos[k, J_PSS_R6_s2]] = KS * dy4_ds2 / T6
    nz[jac_pos[k, J_PSS_R6_s3]] = KS * dy4_ds3 / T6
    nz[jac_pos[k, J_PSS_R6_s4]] = KS * dy4_ds4 / T6
    nz[jac_pos[k, J_PSS_R6_s5]] = KS * dy4_ds5 / T6
    nz[jac_pos[k, J_PSS_R6_s6]] = -1.0 / T6

    # Alg row: f_vs = vs - (T5/T6)*(KS*y4 - s6)
    rT56 = T5/T6
    nz[jac_pos[k, J_PSS_VA_s1]] = -rT56 * KS * dy4_ds1
    nz[jac_pos[k, J_PSS_VA_s2]] = -rT56 * KS * dy4_ds2
    nz[jac_pos[k, J_PSS_VA_s3]] = -rT56 * KS * dy4_ds3
    nz[jac_pos[k, J_PSS_VA_s4]] = -rT56 * KS * dy4_ds4
    nz[jac_pos[k, J_PSS_VA_s5]] = -rT56 * KS * dy4_ds5
    nz[jac_pos[k, J_PSS_VA_s6]] = rT56
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
