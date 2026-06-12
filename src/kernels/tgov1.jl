# TGOV1 batched residual & Jacobian.
#
# States: 2 diff (x1, x2) + 1 alg (p_m). Pref is init-derived (stored in pvec[8]).
# Residual:
#   F[dp+0] = (-x1 + (1 - T2/T3)·x2) / T3
#   F[dp+1] = ((pref - w)/R - x2) / T1
#   F[ap]   = x1 + (T2/T3)·x2 - DT·w - p_m
#
# Jacobian slots (8 total per device):
#   row dp+0: ∂/∂x1, ∂/∂x2
#   row dp+1: ∂/∂x2, ∂/∂w
#   row ap:   ∂/∂x1, ∂/∂x2, ∂/∂w, ∂/∂p_m

const TGOV1_JAC_NENTRIES = 8

const J_TG_R1_x1 = 1
const J_TG_R1_x2 = 2
const J_TG_R2_x2 = 3
const J_TG_R2_w  = 4
const J_TG_A_x1  = 5
const J_TG_A_x2  = 6
const J_TG_A_w   = 7
const J_TG_A_pm  = 8

function tgov1_preallocate!(coord_list::Vector{Vector{Int}},
                             table::TGOV1Table, diff_dim::Int)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        w  = Int(table.w_idx[k])

        # row dp+0
        push!(coord_list[dp],     dp)
        push!(coord_list[dp],     dp + 1)
        # row dp+1
        push!(coord_list[dp + 1], dp + 1)
        if w != 0
            push!(coord_list[dp + 1], w)
        end
        # row ap
        push!(coord_list[ap], dp)
        push!(coord_list[ap], dp + 1)
        if w != 0
            push!(coord_list[ap], w)
        end
        push!(coord_list[ap], ap)
    end
    return nothing
end

function tgov1_jac_positions!(table::TGOV1Table, J::SparseMatrixCSC,
                               diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        w  = Int(table.w_idx[k])

        table.jac_pos[k, J_TG_R1_x1] = _find_pos(J, rows, dp,     dp)
        table.jac_pos[k, J_TG_R1_x2] = _find_pos(J, rows, dp,     dp + 1)
        table.jac_pos[k, J_TG_R2_x2] = _find_pos(J, rows, dp + 1, dp + 1)
        table.jac_pos[k, J_TG_R2_w]  = w == 0 ? Int32(0) : _find_pos(J, rows, dp + 1, w)
        table.jac_pos[k, J_TG_A_x1]  = _find_pos(J, rows, ap, dp)
        table.jac_pos[k, J_TG_A_x2]  = _find_pos(J, rows, ap, dp + 1)
        table.jac_pos[k, J_TG_A_w]   = w == 0 ? Int32(0) : _find_pos(J, rows, ap, w)
        table.jac_pos[k, J_TG_A_pm]  = _find_pos(J, rows, ap, ap)
    end
    return nothing
end

@inline function tgov1_residual_batch!(f::AbstractArray, z::AbstractArray,
                                        p::AbstractArray, table::TGOV1Table,
                                        diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        pp = Int(table.par_ptr[k])
        w_idx = Int(table.w_idx[k])

        R  = p[pp]
        T1 = p[pp + 1]
        T2 = p[pp + 4]
        T3 = p[pp + 5]
        DT = p[pp + 6]
        pref = p[pp + 7]

        x1 = z[dp]
        x2 = z[dp + 1]
        p_m = z[ap]
        w = w_idx == 0 ? 0.0 : z[w_idx]

        t2_over_t3 = T2 / T3
        f[dp]     = (-x1 + (1.0 - t2_over_t3) * x2) / T3
        f[dp + 1] = ((pref - w) / R - x2) / T1
        f[ap]     = x1 + t2_over_t3 * x2 - DT * w - p_m
    end
    return nothing
end

@inline function tgov1_jacobian_batch!(J::SparseMatrixCSC, p::AbstractArray,
                                        table::TGOV1Table, diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        pp = Int(table.par_ptr[k])
        R  = p[pp]
        T1 = p[pp + 1]
        T2 = p[pp + 4]
        T3 = p[pp + 5]
        DT = p[pp + 6]
        t2_over_t3 = T2 / T3

        nz[table.jac_pos[k, J_TG_R1_x1]] = -1.0 / T3
        nz[table.jac_pos[k, J_TG_R1_x2]] = (1.0 - t2_over_t3) / T3

        nz[table.jac_pos[k, J_TG_R2_x2]] = -1.0 / T1
        pos_w2 = table.jac_pos[k, J_TG_R2_w]
        if pos_w2 != 0
            nz[pos_w2] = -1.0 / (R * T1)
        end

        nz[table.jac_pos[k, J_TG_A_x1]] = 1.0
        nz[table.jac_pos[k, J_TG_A_x2]] = t2_over_t3
        pos_wa = table.jac_pos[k, J_TG_A_w]
        if pos_wa != 0
            nz[pos_wa] = -DT
        end
        nz[table.jac_pos[k, J_TG_A_pm]] = -1.0
    end
    return nothing
end
