# ESDC1A batched residual & Jacobian.
#
# States: 3 diff (vr1, vr2, e_fd). No alg. Reads vm = sqrt(vr^2 + vi^2) at its bus.
# Optionally reads v_s from a PSS via vs_idx (0 = no PSS attached, v_s = 0).
# Residual:
#   F[dp]   = (Ka·(vref - vm + vs - vr2 - (Kf/Tf)·e_fd) - vr1) / Ta
#   F[dp+1] = -((Kf/Tf)·e_fd + vr2) / Tf
#   F[dp+2] = (vr1 - Ke·e_fd - Se(e_fd)) / Te
#
# Quadratic saturation:
#   Se(e_fd) = sat_b·(e_fd - sat_a)^2  for e_fd > sat_a, else 0
# sat_a / sat_b are stored in pvec slots 8 / 9.
#
# 11 Jacobian entries per device (9 base + 2 for v_s when PSS attached):
#   row dp:   ∂/∂{vr1, vr2, e_fd, vr, vi, [vs]}  → 5 or 6 entries
#   row dp+1: ∂/∂{vr2, e_fd}                     → 2 entries
#   row dp+2: ∂/∂{vr1, e_fd}                     → 2 entries
# Total: 9 base + 1 for vs in row dp = 10. Actually only row dp
# depends on vm (and hence vs). Rows dp+1 and dp+2 do not.

const ESDC1A_JAC_NENTRIES = 10

const J_EX_R1_vr1 = 1
const J_EX_R1_vr2 = 2
const J_EX_R1_efd = 3
const J_EX_R1_vr  = 4
const J_EX_R1_vi  = 5
const J_EX_R1_vs  = 6
const J_EX_R2_vr2 = 7
const J_EX_R2_efd = 8
const J_EX_R3_vr1 = 9
const J_EX_R3_efd = 10

# --------------------------------------------------------------------
# Sparsity contribution
# --------------------------------------------------------------------

function esdc1a_preallocate!(coord_list::Vector{Vector{Int}},
                              table::ESDC1ATable)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        vr = Int(table.vr_idx[k])
        vi = vr + 1
        vsi = Int(table.vs_idx[k])

        # row dp
        push!(coord_list[dp],     dp)
        push!(coord_list[dp],     dp + 1)
        push!(coord_list[dp],     dp + 2)
        push!(coord_list[dp],     vr)
        push!(coord_list[dp],     vi)
        if vsi > 0
            push!(coord_list[dp], vsi)
        end
        # row dp+1
        push!(coord_list[dp + 1], dp + 1)
        push!(coord_list[dp + 1], dp + 2)
        # row dp+2
        push!(coord_list[dp + 2], dp)
        push!(coord_list[dp + 2], dp + 2)
    end
    return nothing
end

# --------------------------------------------------------------------
# Position cache
# --------------------------------------------------------------------

function esdc1a_jac_positions!(table::ESDC1ATable, J::SparseMatrixCSC)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        vr = Int(table.vr_idx[k])
        vi = vr + 1
        vsi = Int(table.vs_idx[k])

        table.jac_pos[k, J_EX_R1_vr1] = _find_pos(J, rows, dp,     dp)
        table.jac_pos[k, J_EX_R1_vr2] = _find_pos(J, rows, dp,     dp + 1)
        table.jac_pos[k, J_EX_R1_efd] = _find_pos(J, rows, dp,     dp + 2)
        table.jac_pos[k, J_EX_R1_vr]  = _find_pos(J, rows, dp,     vr)
        table.jac_pos[k, J_EX_R1_vi]  = _find_pos(J, rows, dp,     vi)
        table.jac_pos[k, J_EX_R1_vs]  = vsi > 0 ? _find_pos(J, rows, dp, vsi) : Int32(0)
        table.jac_pos[k, J_EX_R2_vr2] = _find_pos(J, rows, dp + 1, dp + 1)
        table.jac_pos[k, J_EX_R2_efd] = _find_pos(J, rows, dp + 1, dp + 2)
        table.jac_pos[k, J_EX_R3_vr1] = _find_pos(J, rows, dp + 2, dp)
        table.jac_pos[k, J_EX_R3_efd] = _find_pos(J, rows, dp + 2, dp + 2)
    end
    return nothing
end

# --------------------------------------------------------------------
# Residual batch
# --------------------------------------------------------------------

@inline function _esdc1a_residual_one!(f, z, p,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr,
        k::Int)
    @inbounds begin
    dp = Int(diff_ptr[k])
    pp = Int(par_ptr[k])
    vr_idx = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

    Ka    = p[pp]
    Ta    = p[pp + 1]
    Kf    = p[pp + 2]
    Tf    = p[pp + 3]
    Ke    = p[pp + 4]
    Te    = p[pp + 5]
    sat_a = p[pp + 7]
    sat_b = p[pp + 8]
    vref  = p[pp + 9]

    vr1  = z[dp]
    vr2  = z[dp + 1]
    e_fd = z[dp + 2]
    vr   = z[vr_idx]
    vi   = z[vr_idx + 1]
    vm   = sqrt(vr*vr + vi*vi)
    vm   = vm == 0.0 ? 1e-12 : vm
    vs   = vsi > 0 ? z[vsi] : 0.0
    sat  = (sat_b == 0.0 || e_fd <= sat_a) ? 0.0 : sat_b*(e_fd - sat_a)^2

    f[dp]     = (Ka*(vref - vm + vs - vr2 - (Kf/Tf)*e_fd) - vr1) / Ta
    f[dp + 1] = -((Kf/Tf)*e_fd + vr2) / Tf
    f[dp + 2] = (vr1 - Ke*e_fd - sat) / Te
    end
    return nothing
end

@inline function esdc1a_residual_batch!(f::AbstractArray, z::AbstractArray,
                                         p::AbstractArray, table::ESDC1ATable)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _esdc1a_residual_one!(f, z, p,
            table.diff_ptr, table.par_ptr, table.vr_idx, table.vs_idx,
            k)
    end
    return nothing
end

# --------------------------------------------------------------------
# Jacobian batch
# --------------------------------------------------------------------

@inline function _esdc1a_jacobian_one!(nz, z, p,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos,
        k::Int)
    @inbounds begin
    pp = Int(par_ptr[k])
    vr_idx = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])
    dp = Int(diff_ptr[k])

    Ka    = p[pp]
    Ta    = p[pp + 1]
    Kf    = p[pp + 2]
    Tf    = p[pp + 3]
    Ke    = p[pp + 4]
    Te    = p[pp + 5]
    sat_a = p[pp + 7]
    sat_b = p[pp + 8]

    e_fd = z[dp + 2]
    vr = z[vr_idx]
    vi = z[vr_idx + 1]
    vm = sqrt(vr*vr + vi*vi)
    vm = vm == 0.0 ? 1e-12 : vm
    dvm_dvr = vr / vm
    dvm_dvi = vi / vm
    dsat = (sat_b == 0.0 || e_fd <= sat_a) ? 0.0 : 2.0*sat_b*(e_fd - sat_a)

    # row dp
    nz[jac_pos[k, J_EX_R1_vr1]] = -1.0 / Ta
    nz[jac_pos[k, J_EX_R1_vr2]] = -Ka / Ta
    nz[jac_pos[k, J_EX_R1_efd]] = -Ka*Kf / (Ta*Tf)
    nz[jac_pos[k, J_EX_R1_vr]]  = -Ka*dvm_dvr / Ta
    nz[jac_pos[k, J_EX_R1_vi]]  = -Ka*dvm_dvi / Ta
    if vsi > 0
        nz[jac_pos[k, J_EX_R1_vs]] = Ka / Ta
    end

    # row dp+1
    nz[jac_pos[k, J_EX_R2_vr2]] = -1.0 / Tf
    nz[jac_pos[k, J_EX_R2_efd]] = -Kf / (Tf*Tf)

    # row dp+2
    nz[jac_pos[k, J_EX_R3_vr1]] = 1.0 / Te
    nz[jac_pos[k, J_EX_R3_efd]] = -(Ke + dsat) / Te
    end
    return nothing
end

@inline function esdc1a_jacobian_batch!(J::SparseMatrixCSC, z::AbstractArray,
                                         p::AbstractArray, table::ESDC1ATable)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _esdc1a_jacobian_one!(nz, z, p,
            table.par_ptr, table.vr_idx, table.vs_idx, table.diff_ptr, table.jac_pos,
            k)
    end
    return nothing
end
