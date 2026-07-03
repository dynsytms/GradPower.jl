# ZIPLoad batched residual & Jacobian.
#
# ZIPLoad is purely algebraic — contributes only to the network voltage
# rows via `cinject!`. No diff/alg states. Math is the verbatim
# translation of `src/loads.jl::cinject!(::ZIPLoad)` and
# `rhs_jac!(::ZIPLoad)`.
#
# Acceptance: parity max|Δf_net| < 1e-14 vs the legacy heterogeneous
# loop on the existing validation cases.

# 4 Jacobian slots per load: vr-row × {vr, vi}, vi-row × {vr, vi}.
const J_ZL_VR_VR = 1
const J_ZL_VR_VI = 2
const J_ZL_VI_VR = 3
const J_ZL_VI_VI = 4

@assert J_ZL_VI_VI == ZIPLOAD_JAC_NENTRIES "Slot table out of sync with ZIPLOAD_JAC_NENTRIES"

# --------------------------------------------------------------------
# Residual batch (current injection — accumulated into voltage rows)
# --------------------------------------------------------------------

"""
    zipload_residual_batch!(f, z, table::ZIPLoadTable, net_ptr::Int)

For each ZIPLoad, ACCUMULATE its current injection into `f[vr]`, `f[vi]`
at the load's bus. Mirrors `cinject!(::ZIPLoad)`.

Loads with v_m² ≤ 0.2 use a saturated constant-power evaluation
(divides by 0.2 instead of v_m²) for numerical stability — same
threshold as the legacy code.
"""
@inline function _zipload_residual_one!(f, z, p,
        bus_arr, par_ptr,
        k::Int, net_ptr::Int,
        inj=nothing, inj_slot::Int=0)
    @inbounds begin
    bus = Int(bus_arr[k])
    pp = Int(par_ptr[k])
    vr_idx = net_ptr + 2*(bus - 1) + 1
    vi_idx = vr_idx + 1
    vr = z[vr_idx]
    vi = z[vi_idx]

    pl = p[pp]            # pinj
    ql = p[pp + 1]        # qinj
    α  = p[pp + 2]
    yreal = α * p[pp + 7] # yreal
    yimag = α * p[pp + 8] # yimag

    vm2 = vr*vr + vi*vi
    vm2_tld = 0.2

    if inj === nothing
        # Plain-loop path: accumulate directly into f (preserves original
        # computation order for bit-identical results).
        f[vr_idx] -= vr*yreal - vi*yimag
        f[vi_idx] -= vr*yimag + vi*yreal

        if vm2 > vm2_tld
            f[vr_idx] -= (1.0 - α) * (pl*vr - ql*vi) / vm2
            f[vi_idx] -= (1.0 - α) * (ql*vr + pl*vi) / vm2
        else
            f[vr_idx] -= (1.0 - α) * (pl*vr - ql*vi) / vm2_tld
            f[vi_idx] -= (1.0 - α) * (ql*vr + pl*vi) / vm2_tld
        end
    else
        # KA path: accumulate into locals, write to injection buffer.
        inj_r = -(vr*yreal - vi*yimag)
        inj_i = -(vr*yimag + vi*yreal)

        if vm2 > vm2_tld
            inj_r -= (1.0 - α) * (pl*vr - ql*vi) / vm2
            inj_i -= (1.0 - α) * (ql*vr + pl*vi) / vm2
        else
            inj_r -= (1.0 - α) * (pl*vr - ql*vi) / vm2_tld
            inj_i -= (1.0 - α) * (ql*vr + pl*vi) / vm2_tld
        end

        inj[2*inj_slot - 1] = inj_r
        inj[2*inj_slot]     = inj_i
    end
    end
    return nothing
end

@inline function zipload_residual_batch!(f::AbstractArray, z::AbstractArray,
                                          p::AbstractArray, table::ZIPLoadTable,
                                          net_ptr::Int)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _zipload_residual_one!(f, z, p,
            table.bus, table.par_ptr,
            k, net_ptr)
    end
    return nothing
end

# --------------------------------------------------------------------
# Position cache + sparsity contribution
# --------------------------------------------------------------------
#
# ZIPLoad's nzval positions already exist in J after the network block
# is built (vr-row and vi-row entries against vr and vi columns of the
# same bus are guaranteed by `preallocate_jacobian`'s adjacency walk —
# the bus is adjacent to itself). So no extra `zipload_preallocate!`
# is needed.

function zipload_jac_positions!(table::ZIPLoadTable, J::SparseMatrixCSC,
                                 net_ptr::Int)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        bus = Int(table.bus[k])
        vr_idx = net_ptr + 2*(bus - 1) + 1
        vi_idx = vr_idx + 1
        table.jac_pos[k, J_ZL_VR_VR] = _find_pos(J, rows, vr_idx, vr_idx)
        table.jac_pos[k, J_ZL_VR_VI] = _find_pos(J, rows, vr_idx, vi_idx)
        table.jac_pos[k, J_ZL_VI_VR] = _find_pos(J, rows, vi_idx, vr_idx)
        table.jac_pos[k, J_ZL_VI_VI] = _find_pos(J, rows, vi_idx, vi_idx)
    end
    @assert all(!iszero, table.jac_pos) "ZIPLoad jac_pos has zero slots — sparsity pattern mismatch"
    return nothing
end

# --------------------------------------------------------------------
# Jacobian batch (accumulates into voltage-row positions)
# --------------------------------------------------------------------

@inline function _zipload_jacobian_one!(nz, z, p,
        bus_arr, par_ptr, jac_pos,
        k::Int, net_ptr::Int)
    @inbounds begin
    bus = Int(bus_arr[k])
    pp = Int(par_ptr[k])
    vr_idx = net_ptr + 2*(bus - 1) + 1
    vi_idx = vr_idx + 1
    vr = z[vr_idx]
    vi = z[vi_idx]

    pl = p[pp]
    ql = p[pp + 1]
    α  = p[pp + 2]
    yreal = p[pp + 7]
    yimag = p[pp + 8]
    vm2 = vr*vr + vi*vi
    vm2_tld = 0.2

    # ---- constant-admittance contribution (Y matrix entries) ----
    d_vr_vr_a = -α * yreal
    d_vr_vi_a =  α * yimag
    d_vi_vr_a = -α * yimag
    d_vi_vi_a = -α * yreal

    # ---- constant-power contribution (depends on v_m^2) ----
    if vm2 > vm2_tld
        inv_vm4 = 1.0 / (vm2 * vm2)
        d_vr_vr_p = (1.0 - α) * ((pl*vr - ql*vi) * 2.0 * vr - pl * vm2) * inv_vm4
        d_vr_vi_p = (1.0 - α) * ((pl*vr - ql*vi) * 2.0 * vi + ql * vm2) * inv_vm4
        d_vi_vr_p = (1.0 - α) * ((ql*vr + pl*vi) * 2.0 * vr - ql * vm2) * inv_vm4
        d_vi_vi_p = (1.0 - α) * ((ql*vr + pl*vi) * 2.0 * vi - pl * vm2) * inv_vm4
    else
        inv_th = 1.0 / vm2_tld
        d_vr_vr_p = (1.0 - α) * (-pl) * inv_th
        d_vr_vi_p = (1.0 - α) * ( ql) * inv_th
        d_vi_vr_p = (1.0 - α) * (-ql) * inv_th
        d_vi_vi_p = (1.0 - α) * (-pl) * inv_th
    end

    nz[jac_pos[k, J_ZL_VR_VR]] += d_vr_vr_a + d_vr_vr_p
    nz[jac_pos[k, J_ZL_VR_VI]] += d_vr_vi_a + d_vr_vi_p
    nz[jac_pos[k, J_ZL_VI_VR]] += d_vi_vr_a + d_vi_vr_p
    nz[jac_pos[k, J_ZL_VI_VI]] += d_vi_vi_a + d_vi_vi_p
    end
    return nothing
end

@inline function zipload_jacobian_batch!(J::SparseMatrixCSC, z::AbstractArray,
                                          p::AbstractArray, table::ZIPLoadTable,
                                          net_ptr::Int)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _zipload_jacobian_one!(nz, z, p,
            table.bus, table.par_ptr, table.jac_pos,
            k, net_ptr)
    end
    return nothing
end
