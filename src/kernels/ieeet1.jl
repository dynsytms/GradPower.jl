# IEEET1 batched residual & Jacobian leaves. Model and parameter layout: see
# the IEEET1 header in src/exciters_psse.jl. Generic batch drivers (loop over
# the StdExcTable, sparsity, jac_pos) live in src/kernels/std_exciters.jl.
#
# States (dp+0..3): vc, vr, xf, e_fd.  pvec (pp+0..12):
#   TR KA TA VRMAX VRMIN KE_eff TE KF TF SWITCH A B vref
# TR/TA/TE/TF arrive floored at IEEET1_TMIN (fill_pvec!), so no branches.
#
#   F[dp+0] = (vt − vc)/TR
#   F[dp+1] = (KA·(vref − vc + vs − xf − (KF/TF)·e_fd) − vr)/TA
#   F[dp+2] = −((KF/TF)·e_fd + xf)/TF
#   F[dp+3] = (vr − KE·e_fd − B·max(e_fd − A, 0)²)/TE
#
# Jacobian slots (12): row order, then column order as in _ieeet1_jac_coords.

const IEEET1_JAC_NENTRIES = 12

const J_T1_R1_vc  = 1
const J_T1_R1_vr  = 2
const J_T1_R1_vi  = 3
const J_T1_R2_vc  = 4
const J_T1_R2_vr  = 5
const J_T1_R2_xf  = 6
const J_T1_R2_efd = 7
const J_T1_R2_vs  = 8
const J_T1_R3_xf  = 9
const J_T1_R3_efd = 10
const J_T1_R4_vr  = 11
const J_T1_R4_efd = 12

# (row, col) for every slot, in slot order. col == 0 → slot unused.
@inline _ieeet1_jac_coords(dp::Int, vr::Int, vsi::Int) = (
    (dp,     dp),     (dp,     vr),     (dp,     vr + 1),
    (dp + 1, dp),     (dp + 1, dp + 1), (dp + 1, dp + 2), (dp + 1, dp + 3), (dp + 1, vsi),
    (dp + 2, dp + 2), (dp + 2, dp + 3),
    (dp + 3, dp + 1), (dp + 3, dp + 3),
)

@inline function _ieeet1_residual_one!(f, z, p,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k::Int)
    @inbounds begin
    dp  = Int(diff_ptr[k])
    pp  = Int(par_ptr[k])
    vri = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

    TR = p[pp];      KA = p[pp + 1]; TA = p[pp + 2]
    KE = p[pp + 5];  TE = p[pp + 6]; KF = p[pp + 7]; TF = p[pp + 8]
    A  = p[pp + 10]; B  = p[pp + 11]; vref = p[pp + 12]

    vc = z[dp]; vr = z[dp + 1]; xf = z[dp + 2]; efd = z[dp + 3]
    vre = z[vri]; vim = z[vri + 1]
    vt = sqrt(vre*vre + vim*vim)
    vs = vsi > 0 ? z[vsi] : 0.0
    kf = KF / TF
    dsat = max(efd - A, 0.0)

    f[dp]     = (vt - vc) / TR
    f[dp + 1] = (KA * (vref - vc + vs - xf - kf * efd) - vr) / TA
    f[dp + 2] = -(kf * efd + xf) / TF
    f[dp + 3] = (vr - KE * efd - B * dsat * dsat) / TE
    end
    return nothing
end

@inline function _ieeet1_jacobian_one!(nz, z, p,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k::Int)
    @inbounds begin
    dp  = Int(diff_ptr[k])
    pp  = Int(par_ptr[k])
    vri = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

    TR = p[pp];      KA = p[pp + 1]; TA = p[pp + 2]
    KE = p[pp + 5];  TE = p[pp + 6]; KF = p[pp + 7]; TF = p[pp + 8]
    A  = p[pp + 10]; B  = p[pp + 11]

    efd = z[dp + 3]
    vre = z[vri]; vim = z[vri + 1]
    vt = sqrt(vre*vre + vim*vim)
    vt = vt == 0.0 ? 1e-12 : vt
    kf = KF / TF

    nz[jac_pos[k, J_T1_R1_vc]]  = -1.0 / TR
    nz[jac_pos[k, J_T1_R1_vr]]  = (vre / vt) / TR
    nz[jac_pos[k, J_T1_R1_vi]]  = (vim / vt) / TR

    nz[jac_pos[k, J_T1_R2_vc]]  = -KA / TA
    nz[jac_pos[k, J_T1_R2_vr]]  = -1.0 / TA
    nz[jac_pos[k, J_T1_R2_xf]]  = -KA / TA
    nz[jac_pos[k, J_T1_R2_efd]] = -KA * kf / TA
    if vsi > 0
        nz[jac_pos[k, J_T1_R2_vs]] = KA / TA
    end

    nz[jac_pos[k, J_T1_R3_xf]]  = -1.0 / TF
    nz[jac_pos[k, J_T1_R3_efd]] = -kf / TF

    nz[jac_pos[k, J_T1_R4_vr]]  = 1.0 / TE
    nz[jac_pos[k, J_T1_R4_efd]] = -(KE + 2.0 * B * max(efd - A, 0.0)) / TE
    end
    return nothing
end

ieeet1_preallocate!(coord_list, t::StdExcTable{IEEET1}) =
    _std_exc_preallocate!(_ieeet1_jac_coords, coord_list, t)
ieeet1_jac_positions!(t::StdExcTable{IEEET1}, J::SparseMatrixCSC) =
    _std_exc_jac_positions!(_ieeet1_jac_coords, t, J)
ieeet1_residual_batch!(f, z, p, t::StdExcTable{IEEET1}) =
    _std_exc_residual_batch!(_ieeet1_residual_one!, f, z, p, t)
ieeet1_jacobian_batch!(J::SparseMatrixCSC, z, p, t::StdExcTable{IEEET1}) =
    _std_exc_jacobian_batch!(_ieeet1_jacobian_one!, J, z, p, t)

attaches_to(::Type{IEEET1}) = Genrou
produces_signals(::Type{IEEET1}) = (
    (target_ctrl_offset = 0,          # Genrou ctrl[0] = e_fd
     source_kind        = :diff_at,
     source_offset      = 3),         # IEEET1 diff[3] = e_fd
)
consumes_signals(::Type{IEEET1}) = ()
