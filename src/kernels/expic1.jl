# EXPIC1 batched residual & Jacobian leaves. Model, simplifications and
# parameter layout: see the EXPIC1 header in src/exciters_psse.jl. Generic
# batch drivers live in src/kernels/std_exciters.jl.
#
# States (dp+0..6): vc, xi, xll, vr, e_fd, y1, vf.
# pvec (pp+k is slot k+1): TR=0 KA=1 TA1=2 TA2=5 TA3=6 TA4=7 KF=10 TF1=11
#   TF2=12 KE_eff=15 TE_eff=16 A=24 B=25 vref=26 VB0=27
# TR/TA3/TA4/TF1/TF2/TE arrive floored at EXPIC1_TMIN (fill_pvec!).
#
#   err  = vref − vc + vs − vf ;  va = xi + KA·TA1·err
#   out1 = xll + (TA2/TA3)·(va − xll)
#   F[dp+0] = (vt − vc)/TR
#   F[dp+1] = KA·err
#   F[dp+2] = (va − xll)/TA3
#   F[dp+3] = (out1 − vr)/TA4
#   F[dp+4] = (vr·VB0 − KE·e_fd − B·max(e_fd − A, 0)²)/TE
#   F[dp+5] = (e_fd − y1)/TF1
#   F[dp+6] = (KF·(e_fd − y1)/TF1 − vf)/TF2

const EXPIC1_JAC_NENTRIES = 24

const J_PI_R1_vc  = 1;  const J_PI_R1_vr  = 2;  const J_PI_R1_vi  = 3
const J_PI_R2_vc  = 4;  const J_PI_R2_vf  = 5;  const J_PI_R2_vs  = 6
const J_PI_R3_xi  = 7;  const J_PI_R3_vc  = 8;  const J_PI_R3_vf  = 9
const J_PI_R3_xll = 10; const J_PI_R3_vs  = 11
const J_PI_R4_xll = 12; const J_PI_R4_xi  = 13; const J_PI_R4_vc  = 14
const J_PI_R4_vf  = 15; const J_PI_R4_vrs = 16; const J_PI_R4_vs  = 17
const J_PI_R5_vrs = 18; const J_PI_R5_efd = 19
const J_PI_R6_efd = 20; const J_PI_R6_y1  = 21
const J_PI_R7_efd = 22; const J_PI_R7_y1  = 23; const J_PI_R7_vf  = 24

# State offsets: vc=0 xi=1 xll=2 vr=3 efd=4 y1=5 vf=6
@inline _expic1_jac_coords(dp::Int, vr::Int, vsi::Int) = (
    (dp,     dp),     (dp,     vr),     (dp,     vr + 1),
    (dp + 1, dp),     (dp + 1, dp + 6), (dp + 1, vsi),
    (dp + 2, dp + 1), (dp + 2, dp),     (dp + 2, dp + 6), (dp + 2, dp + 2), (dp + 2, vsi),
    (dp + 3, dp + 2), (dp + 3, dp + 1), (dp + 3, dp),     (dp + 3, dp + 6), (dp + 3, dp + 3), (dp + 3, vsi),
    (dp + 4, dp + 3), (dp + 4, dp + 4),
    (dp + 5, dp + 4), (dp + 5, dp + 5),
    (dp + 6, dp + 4), (dp + 6, dp + 5), (dp + 6, dp + 6),
)

@inline function _expic1_residual_one!(f, z, p,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k::Int)
    @inbounds begin
    dp  = Int(diff_ptr[k])
    pp  = Int(par_ptr[k])
    vri = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

    TR  = p[pp];      KA  = p[pp + 1];  TA1 = p[pp + 2]
    TA2 = p[pp + 5];  TA3 = p[pp + 6];  TA4 = p[pp + 7]
    KF  = p[pp + 10]; TF1 = p[pp + 11]; TF2 = p[pp + 12]
    KE  = p[pp + 15]; TE  = p[pp + 16]
    A   = p[pp + 24]; B   = p[pp + 25]; vref = p[pp + 26]; vb0 = p[pp + 27]

    vc = z[dp]; xi = z[dp + 1]; xll = z[dp + 2]; vr = z[dp + 3]
    efd = z[dp + 4]; y1 = z[dp + 5]; vf = z[dp + 6]
    vre = z[vri]; vim = z[vri + 1]
    vt = sqrt(vre*vre + vim*vim)
    vs = vsi > 0 ? z[vsi] : 0.0

    err  = vref - vc + vs - vf
    vpi  = xi + KA * TA1 * err
    out1 = xll + (TA2 / TA3) * (vpi - xll)
    dsat = max(efd - A, 0.0)

    f[dp]     = (vt - vc) / TR
    f[dp + 1] = KA * err
    f[dp + 2] = (vpi - xll) / TA3
    f[dp + 3] = (out1 - vr) / TA4
    f[dp + 4] = (vr * vb0 - KE * efd - B * dsat * dsat) / TE
    f[dp + 5] = (efd - y1) / TF1
    f[dp + 6] = (KF * (efd - y1) / TF1 - vf) / TF2
    end
    return nothing
end

@inline function _expic1_jacobian_one!(nz, z, p,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k::Int)
    @inbounds begin
    dp  = Int(diff_ptr[k])
    pp  = Int(par_ptr[k])
    vri = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

    TR  = p[pp];      KA  = p[pp + 1];  TA1 = p[pp + 2]
    TA2 = p[pp + 5];  TA3 = p[pp + 6];  TA4 = p[pp + 7]
    KF  = p[pp + 10]; TF1 = p[pp + 11]; TF2 = p[pp + 12]
    KE  = p[pp + 15]; TE  = p[pp + 16]
    A   = p[pp + 24]; B   = p[pp + 25]; vb0 = p[pp + 27]

    efd = z[dp + 4]
    vre = z[vri]; vim = z[vri + 1]
    vt = sqrt(vre*vre + vim*vim)
    vt = vt == 0.0 ? 1e-12 : vt
    c = KA * TA1
    r = TA2 / TA3

    nz[jac_pos[k, J_PI_R1_vc]]  = -1.0 / TR
    nz[jac_pos[k, J_PI_R1_vr]]  = (vre / vt) / TR
    nz[jac_pos[k, J_PI_R1_vi]]  = (vim / vt) / TR

    nz[jac_pos[k, J_PI_R2_vc]]  = -KA
    nz[jac_pos[k, J_PI_R2_vf]]  = -KA

    nz[jac_pos[k, J_PI_R3_xi]]  = 1.0 / TA3
    nz[jac_pos[k, J_PI_R3_vc]]  = -c / TA3
    nz[jac_pos[k, J_PI_R3_vf]]  = -c / TA3
    nz[jac_pos[k, J_PI_R3_xll]] = -1.0 / TA3

    nz[jac_pos[k, J_PI_R4_xll]] = (1.0 - r) / TA4
    nz[jac_pos[k, J_PI_R4_xi]]  = r / TA4
    nz[jac_pos[k, J_PI_R4_vc]]  = -r * c / TA4
    nz[jac_pos[k, J_PI_R4_vf]]  = -r * c / TA4
    nz[jac_pos[k, J_PI_R4_vrs]] = -1.0 / TA4

    if vsi > 0
        nz[jac_pos[k, J_PI_R2_vs]] = KA
        nz[jac_pos[k, J_PI_R3_vs]] = c / TA3
        nz[jac_pos[k, J_PI_R4_vs]] = r * c / TA4
    end

    nz[jac_pos[k, J_PI_R5_vrs]] = vb0 / TE
    nz[jac_pos[k, J_PI_R5_efd]] = -(KE + 2.0 * B * max(efd - A, 0.0)) / TE

    nz[jac_pos[k, J_PI_R6_efd]] = 1.0 / TF1
    nz[jac_pos[k, J_PI_R6_y1]]  = -1.0 / TF1

    nz[jac_pos[k, J_PI_R7_efd]] = KF / (TF1 * TF2)
    nz[jac_pos[k, J_PI_R7_y1]]  = -KF / (TF1 * TF2)
    nz[jac_pos[k, J_PI_R7_vf]]  = -1.0 / TF2
    end
    return nothing
end

expic1_preallocate!(coord_list, t::StdExcTable{EXPIC1}) =
    _std_exc_preallocate!(_expic1_jac_coords, coord_list, t)
expic1_jac_positions!(t::StdExcTable{EXPIC1}, J::SparseMatrixCSC) =
    _std_exc_jac_positions!(_expic1_jac_coords, t, J)
expic1_residual_batch!(f, z, p, t::StdExcTable{EXPIC1}) =
    _std_exc_residual_batch!(_expic1_residual_one!, f, z, p, t)
expic1_jacobian_batch!(J::SparseMatrixCSC, z, p, t::StdExcTable{EXPIC1}) =
    _std_exc_jacobian_batch!(_expic1_jacobian_one!, J, z, p, t)

attaches_to(::Type{EXPIC1}) = Genrou
produces_signals(::Type{EXPIC1}) = (
    (target_ctrl_offset = 0,          # Genrou ctrl[0] = e_fd
     source_kind        = :diff_at,
     source_offset      = 4),         # EXPIC1 diff[4] = e_fd
)
consumes_signals(::Type{EXPIC1}) = ()
