# SEXS batched residual & Jacobian.
#
# States: 2 diff (x1, e_fd). No alg. Reads vm = sqrt(vr^2 + vi^2) at its bus.
# Optionally reads v_s from a PSS via vs_idx (0 = no PSS attached, v_s = 0).
# Residual:
#   F[dp]   = (-x1 + (1 - TA_TB)·(vref - vm + vs)) / TB
#   F[dp+1] = (-e_fd + K·(x1 + TA_TB·(vref - vm + vs))) / TE
#
# Jacobian slots per device (9 total):
#   row dp:   ∂/∂x1, ∂/∂vr, ∂/∂vi, [∂/∂vs]
#   row dp+1: ∂/∂x1, ∂/∂e_fd, ∂/∂vr, ∂/∂vi, [∂/∂vs]
# The ∂/∂vs slots are only populated when vs_idx > 0.

const SEXS_JAC_NENTRIES = 9

const J_SX_R1_x1  = 1
const J_SX_R1_vr  = 2
const J_SX_R1_vi  = 3
const J_SX_R1_vs  = 4
const J_SX_R2_x1  = 5
const J_SX_R2_efd = 6
const J_SX_R2_vr  = 7
const J_SX_R2_vi  = 8
const J_SX_R2_vs  = 9

function sexs_preallocate!(coord_list::Vector{Vector{Int}},
                            table::SEXSTable)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        vr = Int(table.vr_idx[k])
        vi = vr + 1
        vsi = Int(table.vs_idx[k])

        push!(coord_list[dp],     dp)
        push!(coord_list[dp],     vr)
        push!(coord_list[dp],     vi)
        if vsi > 0
            push!(coord_list[dp], vsi)
        end
        push!(coord_list[dp + 1], dp)
        push!(coord_list[dp + 1], dp + 1)
        push!(coord_list[dp + 1], vr)
        push!(coord_list[dp + 1], vi)
        if vsi > 0
            push!(coord_list[dp + 1], vsi)
        end
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
        vsi = Int(table.vs_idx[k])

        table.jac_pos[k, J_SX_R1_x1]  = _find_pos(J, rows, dp,     dp)
        table.jac_pos[k, J_SX_R1_vr]  = _find_pos(J, rows, dp,     vr)
        table.jac_pos[k, J_SX_R1_vi]  = _find_pos(J, rows, dp,     vi)
        table.jac_pos[k, J_SX_R1_vs]  = vsi > 0 ? _find_pos(J, rows, dp, vsi) : Int32(0)
        table.jac_pos[k, J_SX_R2_x1]  = _find_pos(J, rows, dp + 1, dp)
        table.jac_pos[k, J_SX_R2_efd] = _find_pos(J, rows, dp + 1, dp + 1)
        table.jac_pos[k, J_SX_R2_vr]  = _find_pos(J, rows, dp + 1, vr)
        table.jac_pos[k, J_SX_R2_vi]  = _find_pos(J, rows, dp + 1, vi)
        table.jac_pos[k, J_SX_R2_vs]  = vsi > 0 ? _find_pos(J, rows, dp + 1, vsi) : Int32(0)
    end
    return nothing
end

@inline function _sexs_residual_one!(f, z, p,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr,
        k::Int)
    @inbounds begin
    dp = Int(diff_ptr[k])
    pp = Int(par_ptr[k])
    vr_idx = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

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
    vs   = vsi > 0 ? z[vsi] : 0.0

    err = vref - vm + vs

    f[dp]     = (-x1 + (1.0 - TA_TB)*err) / TB
    f[dp + 1] = (-e_fd + K*(x1 + TA_TB*err)) / TE
    end
    return nothing
end

@inline function sexs_residual_batch!(f::AbstractArray, z::AbstractArray,
                                       p::AbstractArray, table::SEXSTable)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _sexs_residual_one!(f, z, p,
            table.diff_ptr, table.par_ptr, table.vr_idx, table.vs_idx,
            k)
    end
    return nothing
end

@inline function _sexs_jacobian_one!(nz, z, p,
        par_ptr, vr_idx_arr, vs_idx_arr, jac_pos,
        k::Int)
    @inbounds begin
    pp = Int(par_ptr[k])
    vr_idx = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

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
    nz[jac_pos[k, J_SX_R1_x1]] = -1.0 / TB
    nz[jac_pos[k, J_SX_R1_vr]] = -(1.0 - TA_TB) * dvm_dvr / TB
    nz[jac_pos[k, J_SX_R1_vi]] = -(1.0 - TA_TB) * dvm_dvi / TB
    if vsi > 0
        nz[jac_pos[k, J_SX_R1_vs]] = (1.0 - TA_TB) / TB
    end

    # row dp+1:
    nz[jac_pos[k, J_SX_R2_x1]]  = K / TE
    nz[jac_pos[k, J_SX_R2_efd]] = -1.0 / TE
    nz[jac_pos[k, J_SX_R2_vr]]  = -K * TA_TB * dvm_dvr / TE
    nz[jac_pos[k, J_SX_R2_vi]]  = -K * TA_TB * dvm_dvi / TE
    if vsi > 0
        nz[jac_pos[k, J_SX_R2_vs]] = K * TA_TB / TE
    end
    end
    return nothing
end

@inline function sexs_jacobian_batch!(J::SparseMatrixCSC, z::AbstractArray,
                                       p::AbstractArray, table::SEXSTable)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _sexs_jacobian_one!(nz, z, p,
            table.par_ptr, table.vr_idx, table.vs_idx, table.jac_pos,
            k)
    end
    return nothing
end
