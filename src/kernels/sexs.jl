# SEXS batched residual & Jacobian.
#
# States: 2 diff (x1, e_fd). No alg. Reads vm = sqrt(vr^2 + vi^2) at its bus.
# Residual:
#   F[dp]   = (-x1 + (1 - TA_TB)·(vref - vm)) / TB
#   F[dp+1] = (-e_fd + K·(x1 + TA_TB·(vref - vm))) / TE
#
# Jacobian slots per device (7 total):
#   row dp:   ∂/∂x1, ∂/∂vr, ∂/∂vi
#   row dp+1: ∂/∂x1, ∂/∂e_fd, ∂/∂vr, ∂/∂vi

const SEXS_JAC_NENTRIES = 7

const J_SX_R1_x1  = 1
const J_SX_R1_vr  = 2
const J_SX_R1_vi  = 3
const J_SX_R2_x1  = 4
const J_SX_R2_efd = 5
const J_SX_R2_vr  = 6
const J_SX_R2_vi  = 7

function sexs_preallocate!(coord_list::Vector{Vector{Int}},
                            table::SEXSTable)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        vr = Int(table.vr_idx[k])
        vi = vr + 1

        push!(coord_list[dp],     dp)
        push!(coord_list[dp],     vr)
        push!(coord_list[dp],     vi)
        push!(coord_list[dp + 1], dp)
        push!(coord_list[dp + 1], dp + 1)
        push!(coord_list[dp + 1], vr)
        push!(coord_list[dp + 1], vi)
    end
    return nothing
end

function sexs_jac_positions!(table::SEXSTable, J::SparseMatrixCSC)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        vr = Int(table.vr_idx[k])
        vi = vr + 1

        table.jac_pos[k, J_SX_R1_x1]  = _find_pos(J, rows, dp,     dp)
        table.jac_pos[k, J_SX_R1_vr]  = _find_pos(J, rows, dp,     vr)
        table.jac_pos[k, J_SX_R1_vi]  = _find_pos(J, rows, dp,     vi)
        table.jac_pos[k, J_SX_R2_x1]  = _find_pos(J, rows, dp + 1, dp)
        table.jac_pos[k, J_SX_R2_efd] = _find_pos(J, rows, dp + 1, dp + 1)
        table.jac_pos[k, J_SX_R2_vr]  = _find_pos(J, rows, dp + 1, vr)
        table.jac_pos[k, J_SX_R2_vi]  = _find_pos(J, rows, dp + 1, vi)
    end
    return nothing
end

@inline function sexs_residual_batch!(f::AbstractArray, z::AbstractArray,
                                       p::AbstractArray, table::SEXSTable)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        pp = Int(table.par_ptr[k])
        vr_idx = Int(table.vr_idx[k])

        TA_TB = p[pp]
        TB    = p[pp + 1]
        K     = p[pp + 2]
        TE    = p[pp + 3]
        vref  = p[pp + 6]

        x1   = z[dp]
        e_fd = z[dp + 1]
        vr   = z[vr_idx]
        vi   = z[vr_idx + 1]
        vm   = sqrt(vr*vr + vi*vi)
        vm   = vm == 0.0 ? 1e-12 : vm

        f[dp]     = (-x1 + (1.0 - TA_TB)*(vref - vm)) / TB
        f[dp + 1] = (-e_fd + K*(x1 + TA_TB*(vref - vm))) / TE
    end
    return nothing
end

@inline function sexs_jacobian_batch!(J::SparseMatrixCSC, z::AbstractArray,
                                       p::AbstractArray, table::SEXSTable)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        pp = Int(table.par_ptr[k])
        vr_idx = Int(table.vr_idx[k])

        TA_TB = p[pp]
        TB    = p[pp + 1]
        K     = p[pp + 2]
        TE    = p[pp + 3]

        vr = z[vr_idx]
        vi = z[vr_idx + 1]
        vm = sqrt(vr*vr + vi*vi)
        vm = vm == 0.0 ? 1e-12 : vm
        dvm_dvr = vr / vm
        dvm_dvi = vi / vm

        # row dp:
        nz[table.jac_pos[k, J_SX_R1_x1]] = -1.0 / TB
        nz[table.jac_pos[k, J_SX_R1_vr]] = -(1.0 - TA_TB) * dvm_dvr / TB
        nz[table.jac_pos[k, J_SX_R1_vi]] = -(1.0 - TA_TB) * dvm_dvi / TB

        # row dp+1:
        nz[table.jac_pos[k, J_SX_R2_x1]]  = K / TE
        nz[table.jac_pos[k, J_SX_R2_efd]] = -1.0 / TE
        nz[table.jac_pos[k, J_SX_R2_vr]]  = -K * TA_TB * dvm_dvr / TE
        nz[table.jac_pos[k, J_SX_R2_vi]]  = -K * TA_TB * dvm_dvi / TE
    end
    return nothing
end
