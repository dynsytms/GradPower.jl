# ESST4B batched residual & Jacobian.
#
# States: 4 diff (vc, xr, xm, e_fd). No alg. Reads vt = sqrt(vr^2 + vi^2) at
# its bus via vr_idx, and optionally v_s from a PSS via vs_idx (0 = no PSS).
#
# With  err = vref - vc + vs,  vr_pi = xr + KPR*err,
#       vm_pi = xm + KPM*(vr_pi - KG*e_fd):
#
#   F[dp+0] = (vt - vc) / TR
#   F[dp+1] = KIR * err
#   F[dp+2] = KIM * (vr_pi - KG*e_fd)
#   F[dp+3] = (vm_pi*vb0 - e_fd) / TA
#
# vb0 = KP·|Vt(0)| is the potential-source bridge voltage FROZEN at its
# initialization value (pvec slot 19). It is deliberately not live — see
# simplification (2) in the model header in src/exciters.jl for the measured
# instability a live VB = KP·|Vt(t)| causes without the VMMAX/VBMAX limiters.
#
# TR/TA arrive already floored at ESST4B_TMIN (see fill_pvec! / the table
# builder), so there is no branch on T == 0 here. Limiters
# (VRMAX/VRMIN/VMMAX/VMMIN/VBMAX) and the rectifier-loading /
# current-compounding terms (KC/KI/XL/THETAP) are parsed but NOT applied.
#
# Jacobian slots per device (14 total):
#   row dp+0 (vc):   d/dvc, d/dvr, d/dvi
#   row dp+1 (xr):   d/dvc, [d/dvs]                   (no diagonal: dF/dxr == 0)
#   row dp+2 (xm):   d/dxr, d/dvc, d/de_fd, [d/dvs]   (no diagonal)
#   row dp+3 (e_fd): d/dxm, d/dxr, d/dvc, d/de_fd, [d/dvs]
# The backward-Euler assembly adds the 1/dt diagonal for every differential
# row independently (see `preallocate_jacobian` in src/dynamics.jl), so the
# structurally-zero diagonals of rows dp+1 / dp+2 are fine.
# The d/dvs slots are only populated when vs_idx > 0, i.e. when an IEEEST
# is attached and GradPower.ESST4B_WIRE_PSS[] is true (the default; see
# src/exciters.jl).

const ESST4B_JAC_NENTRIES = 14

const J_S4_R1_vc  = 1
const J_S4_R1_vr  = 2
const J_S4_R1_vi  = 3
const J_S4_R2_vc  = 4
const J_S4_R2_vs  = 5
const J_S4_R3_xr  = 6
const J_S4_R3_vc  = 7
const J_S4_R3_efd = 8
const J_S4_R3_vs  = 9
const J_S4_R4_xm  = 10
const J_S4_R4_xr  = 11
const J_S4_R4_vc  = 12
const J_S4_R4_efd = 13
const J_S4_R4_vs  = 14

function esst4b_preallocate!(coord_list::Vector{Vector{Int}},
                              table::ESST4BTable)
    for k in 1:table.n
        dp  = Int(table.diff_ptr[k])
        vr  = Int(table.vr_idx[k])
        vi  = vr + 1
        vsi = Int(table.vs_idx[k])

        # row dp+0 : vc
        push!(coord_list[dp], dp)
        push!(coord_list[dp], vr)
        push!(coord_list[dp], vi)

        # row dp+1 : xr
        push!(coord_list[dp + 1], dp)
        vsi > 0 && push!(coord_list[dp + 1], vsi)

        # row dp+2 : xm
        push!(coord_list[dp + 2], dp + 1)
        push!(coord_list[dp + 2], dp)
        push!(coord_list[dp + 2], dp + 3)
        vsi > 0 && push!(coord_list[dp + 2], vsi)

        # row dp+3 : e_fd
        push!(coord_list[dp + 3], dp + 2)
        push!(coord_list[dp + 3], dp + 1)
        push!(coord_list[dp + 3], dp)
        push!(coord_list[dp + 3], dp + 3)
        vsi > 0 && push!(coord_list[dp + 3], vsi)
    end
    return nothing
end

function esst4b_jac_positions!(table::ESST4BTable, J::SparseMatrixCSC)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        dp  = Int(table.diff_ptr[k])
        vr  = Int(table.vr_idx[k])
        vi  = vr + 1
        vsi = Int(table.vs_idx[k])

        table.jac_pos[k, J_S4_R1_vc]  = _find_pos(J, rows, dp,     dp)
        table.jac_pos[k, J_S4_R1_vr]  = _find_pos(J, rows, dp,     vr)
        table.jac_pos[k, J_S4_R1_vi]  = _find_pos(J, rows, dp,     vi)

        table.jac_pos[k, J_S4_R2_vc]  = _find_pos(J, rows, dp + 1, dp)
        table.jac_pos[k, J_S4_R2_vs]  = vsi > 0 ? _find_pos(J, rows, dp + 1, vsi) : Int32(0)

        table.jac_pos[k, J_S4_R3_xr]  = _find_pos(J, rows, dp + 2, dp + 1)
        table.jac_pos[k, J_S4_R3_vc]  = _find_pos(J, rows, dp + 2, dp)
        table.jac_pos[k, J_S4_R3_efd] = _find_pos(J, rows, dp + 2, dp + 3)
        table.jac_pos[k, J_S4_R3_vs]  = vsi > 0 ? _find_pos(J, rows, dp + 2, vsi) : Int32(0)

        table.jac_pos[k, J_S4_R4_xm]  = _find_pos(J, rows, dp + 3, dp + 2)
        table.jac_pos[k, J_S4_R4_xr]  = _find_pos(J, rows, dp + 3, dp + 1)
        table.jac_pos[k, J_S4_R4_vc]  = _find_pos(J, rows, dp + 3, dp)
        table.jac_pos[k, J_S4_R4_efd] = _find_pos(J, rows, dp + 3, dp + 3)
        table.jac_pos[k, J_S4_R4_vs]  = vsi > 0 ? _find_pos(J, rows, dp + 3, vsi) : Int32(0)
    end
    return nothing
end

@inline function _esst4b_residual_one!(f, z, p,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr,
        k::Int)
    @inbounds begin
    dp     = Int(diff_ptr[k])
    pp     = Int(par_ptr[k])
    vr_idx = Int(vr_idx_arr[k])
    vsi    = Int(vs_idx_arr[k])

    TR   = p[pp]
    KPR  = p[pp + 1]
    KIR  = p[pp + 2]
    TA   = p[pp + 5]
    KPM  = p[pp + 6]
    KIM  = p[pp + 7]
    KG   = p[pp + 10]
    vref = p[pp + 17]
    vb0  = p[pp + 18]

    vc   = z[dp]
    xr   = z[dp + 1]
    xm   = z[dp + 2]
    e_fd = z[dp + 3]

    vr = z[vr_idx]
    vi = z[vr_idx + 1]
    vt = sqrt(vr*vr + vi*vi)
    vt = vt == 0.0 ? 1e-12 : vt
    vs = vsi > 0 ? z[vsi] : 0.0

    err   = vref - vc + vs
    vr_pi = xr + KPR * err
    vm_pi = xm + KPM * (vr_pi - KG * e_fd)

    f[dp]     = (vt - vc) / TR
    f[dp + 1] = KIR * err
    f[dp + 2] = KIM * (vr_pi - KG * e_fd)
    f[dp + 3] = (vm_pi * vb0 - e_fd) / TA
    end
    return nothing
end

@inline function esst4b_residual_batch!(f::AbstractArray, z::AbstractArray,
                                         p::AbstractArray, table::ESST4BTable)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _esst4b_residual_one!(f, z, p,
            table.diff_ptr, table.par_ptr, table.vr_idx, table.vs_idx,
            k)
    end
    return nothing
end

# `diff_ptr` is unused now that row dp+3 is linear in the states, but the
# argument is kept so the signature matches the KA / CUDA wrappers (and
# ESDC1A's leaf) — changing it would ripple into both extension files.
@inline function _esst4b_jacobian_one!(nz, z, p,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos,
        k::Int)
    @inbounds begin
    pp     = Int(par_ptr[k])
    vr_idx = Int(vr_idx_arr[k])
    vsi    = Int(vs_idx_arr[k])

    TR   = p[pp]
    KPR  = p[pp + 1]
    KIR  = p[pp + 2]
    TA   = p[pp + 5]
    KPM  = p[pp + 6]
    KIM  = p[pp + 7]
    KG   = p[pp + 10]
    vb0  = p[pp + 18]

    vr = z[vr_idx]
    vi = z[vr_idx + 1]
    vt = sqrt(vr*vr + vi*vi)
    vt = vt == 0.0 ? 1e-12 : vt
    dvt_dvr = vr / vt
    dvt_dvi = vi / vt

    # row dp+0 : (vt - vc)/TR
    nz[jac_pos[k, J_S4_R1_vc]] = -1.0 / TR
    nz[jac_pos[k, J_S4_R1_vr]] = dvt_dvr / TR
    nz[jac_pos[k, J_S4_R1_vi]] = dvt_dvi / TR

    # row dp+1 : KIR*err
    nz[jac_pos[k, J_S4_R2_vc]] = -KIR
    if vsi > 0
        nz[jac_pos[k, J_S4_R2_vs]] = KIR
    end

    # row dp+2 : KIM*(vr_pi - KG*e_fd)
    nz[jac_pos[k, J_S4_R3_xr]]  = KIM
    nz[jac_pos[k, J_S4_R3_vc]]  = -KIM * KPR
    nz[jac_pos[k, J_S4_R3_efd]] = -KIM * KG
    if vsi > 0
        nz[jac_pos[k, J_S4_R3_vs]] = KIM * KPR
    end

    # row dp+3 : (vm_pi*vb0 - e_fd)/TA
    nz[jac_pos[k, J_S4_R4_xm]]  = vb0 / TA
    nz[jac_pos[k, J_S4_R4_xr]]  = KPM * vb0 / TA
    nz[jac_pos[k, J_S4_R4_vc]]  = -KPM * KPR * vb0 / TA
    nz[jac_pos[k, J_S4_R4_efd]] = (-KPM * KG * vb0 - 1.0) / TA
    if vsi > 0
        nz[jac_pos[k, J_S4_R4_vs]] = KPM * KPR * vb0 / TA
    end
    end
    return nothing
end

@inline function esst4b_jacobian_batch!(J::SparseMatrixCSC, z::AbstractArray,
                                         p::AbstractArray, table::ESST4BTable)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _esst4b_jacobian_one!(nz, z, p,
            table.par_ptr, table.vr_idx, table.vs_idx, table.diff_ptr,
            table.jac_pos, k)
    end
    return nothing
end
