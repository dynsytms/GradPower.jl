# IEESGO batched residual & Jacobian.
#
# IEESGO has 5 diff states (PF0, PLL, TP1, TP2, TP3) + 1 alg state (p_m).
# Residual:
#   F[dp+0] = (1/T1)(K1·w − PF0)
#   F[dp+1] = (1/T3)((1 − T2/T3)·PF0 − PLL)
#   F[dp+2] = (1/T4)(pref − (T2/T3)·PF0 − PLL − TP1)
#   F[dp+3] = (1/T5)(K2·TP1 − TP2)
#   F[dp+4] = (1/T6)(K3·TP2 − TP3)
#   F[ap]   = TP1·(1 − K2) + TP2·(1 − K3) + TP3 − p_m
#
# The legacy GradPower IEESGO has NO rhs_fun!/rhs_jac!/cinject! methods —
# the heterogeneous loop hits the `@warn` fallbacks. So this kernel adds
# behavior that wasn't there; there is no Julia-side parity to assert.
# Acceptance is f(z0) ≈ 0 after `initialize_dynamics!`.

# 15 Jacobian entries per IEESGO device:
#   row dp+0:   d/d{PF0, w}         → 2 entries
#   row dp+1:   d/d{PF0, PLL}       → 2 entries
#   row dp+2:   d/d{PF0, PLL, TP1}  → 3 entries
#   row dp+3:   d/d{TP1, TP2}       → 2 entries
#   row dp+4:   d/d{TP2, TP3}       → 2 entries
#   row ap:     d/d{TP1, TP2, TP3, p_m}  → 4 entries
const IEESGO_JAC_NENTRIES = 15

# Slot layout (must agree across positions + jacobian batch).
const J_IG_R1_PF0  = 1
const J_IG_R1_w    = 2
const J_IG_R2_PF0  = 3
const J_IG_R2_PLL  = 4
const J_IG_R3_PF0  = 5
const J_IG_R3_PLL  = 6
const J_IG_R3_TP1  = 7
const J_IG_R4_TP1  = 8
const J_IG_R4_TP2  = 9
const J_IG_R5_TP2  = 10
const J_IG_R5_TP3  = 11
const J_IG_A_TP1   = 12
const J_IG_A_TP2   = 13
const J_IG_A_TP3   = 14
const J_IG_A_pm    = 15

# --------------------------------------------------------------------
# Sparsity contribution — added to `preallocate_jacobian`'s coord_list.
# --------------------------------------------------------------------

"""
    ieesgo_preallocate!(coord_list, table::IEESGOTable, diff_dim::Int)

Append IEESGO sparsity entries (rows × cols) to `coord_list` so the
resulting `SparseMatrixCSC` has slots for `ieesgo_jacobian_batch!` to
write into. Called once during `preallocate_jacobian`. The `w_idx`
column for row dp+0 is read from the table (set by `wire_controls!`).
"""
function ieesgo_preallocate!(coord_list::Vector{Vector{Int}},
                              table::IEESGOTable, diff_dim::Int)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim   # global alg row
        w  = Int(table.w_idx[k])

        # row dp+0
        push!(coord_list[dp],     dp)            # ∂/∂PF0
        if w != 0
            push!(coord_list[dp], w)             # ∂/∂w
        end
        # row dp+1
        push!(coord_list[dp + 1], dp)
        push!(coord_list[dp + 1], dp + 1)
        # row dp+2
        push!(coord_list[dp + 2], dp)
        push!(coord_list[dp + 2], dp + 1)
        push!(coord_list[dp + 2], dp + 2)
        # row dp+3
        push!(coord_list[dp + 3], dp + 2)
        push!(coord_list[dp + 3], dp + 3)
        # row dp+4
        push!(coord_list[dp + 4], dp + 3)
        push!(coord_list[dp + 4], dp + 4)
        # row ap (global)
        push!(coord_list[ap], dp + 2)
        push!(coord_list[ap], dp + 3)
        push!(coord_list[ap], dp + 4)
        push!(coord_list[ap], ap)
    end
    return nothing
end

# --------------------------------------------------------------------
# Position cache
# --------------------------------------------------------------------

function ieesgo_jac_positions!(table::IEESGOTable, J::SparseMatrixCSC,
                                diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        w  = Int(table.w_idx[k])

        table.jac_pos[k, J_IG_R1_PF0] = _find_pos(J, rows, dp,     dp)
        # w_idx may legitimately be 0 (orphan governor, e.g. synthetic
        # case). Leave the slot at 0 — the kernel skips it when w == 0.
        table.jac_pos[k, J_IG_R1_w]   = w == 0 ? Int32(0) : _find_pos(J, rows, dp, w)

        table.jac_pos[k, J_IG_R2_PF0] = _find_pos(J, rows, dp + 1, dp)
        table.jac_pos[k, J_IG_R2_PLL] = _find_pos(J, rows, dp + 1, dp + 1)

        table.jac_pos[k, J_IG_R3_PF0] = _find_pos(J, rows, dp + 2, dp)
        table.jac_pos[k, J_IG_R3_PLL] = _find_pos(J, rows, dp + 2, dp + 1)
        table.jac_pos[k, J_IG_R3_TP1] = _find_pos(J, rows, dp + 2, dp + 2)

        table.jac_pos[k, J_IG_R4_TP1] = _find_pos(J, rows, dp + 3, dp + 2)
        table.jac_pos[k, J_IG_R4_TP2] = _find_pos(J, rows, dp + 3, dp + 3)

        table.jac_pos[k, J_IG_R5_TP2] = _find_pos(J, rows, dp + 4, dp + 3)
        table.jac_pos[k, J_IG_R5_TP3] = _find_pos(J, rows, dp + 4, dp + 4)

        table.jac_pos[k, J_IG_A_TP1]  = _find_pos(J, rows, ap, dp + 2)
        table.jac_pos[k, J_IG_A_TP2]  = _find_pos(J, rows, ap, dp + 3)
        table.jac_pos[k, J_IG_A_TP3]  = _find_pos(J, rows, ap, dp + 4)
        table.jac_pos[k, J_IG_A_pm]   = _find_pos(J, rows, ap, ap)
    end
    return nothing
end

# --------------------------------------------------------------------
# Residual batch
# --------------------------------------------------------------------

@inline function _ieesgo_residual_one!(f, z, p,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        k::Int, diff_dim::Int)
    @inbounds begin
    dp = Int(diff_ptr[k])
    ap = Int(alg_ptr[k]) + diff_dim
    pp = Int(par_ptr[k])
    w_idx = Int(w_idx_arr[k])

    T1 = p[pp];      T2 = p[pp + 1]; T3 = p[pp + 2]
    T4 = p[pp + 3];  T5 = p[pp + 4]; T6 = p[pp + 5]
    K1 = p[pp + 6];  K2 = p[pp + 7]; K3 = p[pp + 8]
    pref = p[pp + 11]

    PF0 = z[dp]
    PLL = z[dp + 1]
    TP1 = z[dp + 2]
    TP2 = z[dp + 3]
    TP3 = z[dp + 4]
    p_m = z[ap]
    w = w_idx == 0 ? 0.0 : z[w_idx]

    f[dp]     = (1.0 / T1) * (K1 * w - PF0)
    f[dp + 1] = (1.0 / T3) * ((1.0 - (T2 / T3)) * PF0 - PLL)
    SatP = pref - (T2 / T3) * PF0 - PLL
    f[dp + 2] = (1.0 / T4) * (SatP - TP1)
    f[dp + 3] = (1.0 / T5) * (K2 * TP1 - TP2)
    f[dp + 4] = (1.0 / T6) * (K3 * TP2 - TP3)
    f[ap]     = TP1 * (1.0 - K2) + TP2 * (1.0 - K3) + TP3 - p_m
    end
    return nothing
end

@inline function ieesgo_residual_batch!(f::AbstractArray, z::AbstractArray,
                                         p::AbstractArray, table::IEESGOTable,
                                         diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _ieesgo_residual_one!(f, z, p,
            table.diff_ptr, table.alg_ptr, table.par_ptr, table.w_idx,
            k, diff_dim)
    end
    return nothing
end

# --------------------------------------------------------------------
# Jacobian batch
# --------------------------------------------------------------------

@inline function _ieesgo_jacobian_one!(nz, p,
        par_ptr, jac_pos,
        k::Int)
    @inbounds begin
    pp = Int(par_ptr[k])
    T1 = p[pp];      T2 = p[pp + 1]; T3 = p[pp + 2]
    T4 = p[pp + 3];  T5 = p[pp + 4]; T6 = p[pp + 5]
    K1 = p[pp + 6];  K2 = p[pp + 7]; K3 = p[pp + 8]

    # row dp+0: d/dPF0 = -1/T1, d/dw = K1/T1
    nz[jac_pos[k, J_IG_R1_PF0]] = -1.0 / T1
    pos_w = jac_pos[k, J_IG_R1_w]
    if pos_w != 0
        nz[pos_w] = K1 / T1
    end

    # row dp+1: d/dPF0 = (1 - T2/T3)/T3, d/dPLL = -1/T3
    nz[jac_pos[k, J_IG_R2_PF0]] = (1.0 - T2 / T3) / T3
    nz[jac_pos[k, J_IG_R2_PLL]] = -1.0 / T3

    # row dp+2: d/dPF0 = -T2/(T3*T4), d/dPLL = -1/T4, d/dTP1 = -1/T4
    nz[jac_pos[k, J_IG_R3_PF0]] = -T2 / (T3 * T4)
    nz[jac_pos[k, J_IG_R3_PLL]] = -1.0 / T4
    nz[jac_pos[k, J_IG_R3_TP1]] = -1.0 / T4

    # row dp+3: d/dTP1 = K2/T5, d/dTP2 = -1/T5
    nz[jac_pos[k, J_IG_R4_TP1]] = K2 / T5
    nz[jac_pos[k, J_IG_R4_TP2]] = -1.0 / T5

    # row dp+4: d/dTP2 = K3/T6, d/dTP3 = -1/T6
    nz[jac_pos[k, J_IG_R5_TP2]] = K3 / T6
    nz[jac_pos[k, J_IG_R5_TP3]] = -1.0 / T6

    # row ap: d/dTP1 = 1-K2, d/dTP2 = 1-K3, d/dTP3 = 1, d/dp_m = -1
    nz[jac_pos[k, J_IG_A_TP1]] = 1.0 - K2
    nz[jac_pos[k, J_IG_A_TP2]] = 1.0 - K3
    nz[jac_pos[k, J_IG_A_TP3]] = 1.0
    nz[jac_pos[k, J_IG_A_pm]]  = -1.0
    end
    return nothing
end

@inline function ieesgo_jacobian_batch!(J::SparseMatrixCSC, p::AbstractArray,
                                         table::IEESGOTable, diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _ieesgo_jacobian_one!(nz, p,
            table.par_ptr, table.jac_pos,
            k)
    end
    return nothing
end
