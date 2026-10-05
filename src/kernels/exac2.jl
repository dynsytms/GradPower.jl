# EXAC2 batched residual & Jacobian leaves. Model, simplifications and
# parameter layout: see the EXAC2 header in src/exciters_psse.jl. Generic
# batch drivers live in src/kernels/std_exciters.jl.
#
# States (dp+0..4): vc, xll, va, e_fd, xf.
# pvec (pp+k is slot k+1): TR=0 TB=1 TC=2 KA=3 TA=4 KB=7 TE=10 KH=12 KF=13
#   TF=14 KD=16 KE=17 A=23 B=24 kfex=25 vref=26
# TR/TB/TA/TE/TF arrive floored at EXAC2_TMIN (fill_pvec!).
#
#   VE  = kfex·e_fd ;  VFE = KE·VE + B·max(VE − A, 0)² + KD·e_fd
#   vf  = xf + (KF/TF)·VFE ;  u = vref − vc + vs − vf
#   out = xll + (TC/TB)·(u − xll) ;  VR = KB·(va − KH·VFE)
#   F[dp+0] = (vt − vc)/TR
#   F[dp+1] = (u − xll)/TB
#   F[dp+2] = (KA·out − va)/TA
#   F[dp+3] = (VR − VFE)/(TE·kfex)
#   F[dp+4] = −((KF/TF)·VFE + xf)/TF
# with g = dVFE/de_fd = kfex·(KE + 2B·max(VE − A, 0)) + KD.

const EXAC2_JAC_NENTRIES = 18

const J_A2_R1_vc  = 1;  const J_A2_R1_vr  = 2;  const J_A2_R1_vi  = 3
const J_A2_R2_vc  = 4;  const J_A2_R2_xf  = 5;  const J_A2_R2_efd = 6
const J_A2_R2_xll = 7;  const J_A2_R2_vs  = 8
const J_A2_R3_xll = 9;  const J_A2_R3_vc  = 10; const J_A2_R3_xf  = 11
const J_A2_R3_efd = 12; const J_A2_R3_va  = 13; const J_A2_R3_vs  = 14
const J_A2_R4_va  = 15; const J_A2_R4_efd = 16
const J_A2_R5_efd = 17; const J_A2_R5_xf  = 18

# State offsets: vc=0 xll=1 va=2 efd=3 xf=4
@inline _exac2_jac_coords(dp::Int, vr::Int, vsi::Int) = (
    (dp,     dp),     (dp,     vr),     (dp,     vr + 1),
    (dp + 1, dp),     (dp + 1, dp + 4), (dp + 1, dp + 3), (dp + 1, dp + 1), (dp + 1, vsi),
    (dp + 2, dp + 1), (dp + 2, dp),     (dp + 2, dp + 4), (dp + 2, dp + 3), (dp + 2, dp + 2), (dp + 2, vsi),
    (dp + 3, dp + 2), (dp + 3, dp + 3),
    (dp + 4, dp + 3), (dp + 4, dp + 4),
)

@inline function _exac2_residual_one!(f, z, p,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k::Int)
    @inbounds begin
    dp  = Int(diff_ptr[k])
    pp  = Int(par_ptr[k])
    vri = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

    TR = p[pp];      TB = p[pp + 1];  TC = p[pp + 2]; KA = p[pp + 3]
    TA = p[pp + 4];  KB = p[pp + 7];  TE = p[pp + 10]; KH = p[pp + 12]
    KF = p[pp + 13]; TF = p[pp + 14]; KD = p[pp + 16]; KE = p[pp + 17]
    A  = p[pp + 23]; B  = p[pp + 24]; kfex = p[pp + 25]; vref = p[pp + 26]

    vc = z[dp]; xll = z[dp + 1]; vA = z[dp + 2]; efd = z[dp + 3]; xf = z[dp + 4]
    vre = z[vri]; vim = z[vri + 1]
    vt = sqrt(vre*vre + vim*vim)
    vs = vsi > 0 ? z[vsi] : 0.0

    kf   = KF / TF
    ve   = kfex * efd
    dsat = max(ve - A, 0.0)
    vfe  = KE * ve + B * dsat * dsat + KD * efd
    u    = vref - vc + vs - (xf + kf * vfe)
    out  = xll + (TC / TB) * (u - xll)

    f[dp]     = (vt - vc) / TR
    f[dp + 1] = (u - xll) / TB
    f[dp + 2] = (KA * out - vA) / TA
    f[dp + 3] = (KB * (vA - KH * vfe) - vfe) / (TE * kfex)
    f[dp + 4] = -(kf * vfe + xf) / TF
    end
    return nothing
end

@inline function _exac2_jacobian_one!(nz, z, p,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k::Int)
    @inbounds begin
    dp  = Int(diff_ptr[k])
    pp  = Int(par_ptr[k])
    vri = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

    TR = p[pp];      TB = p[pp + 1];  TC = p[pp + 2]; KA = p[pp + 3]
    TA = p[pp + 4];  KB = p[pp + 7];  TE = p[pp + 10]; KH = p[pp + 12]
    KF = p[pp + 13]; TF = p[pp + 14]; KD = p[pp + 16]; KE = p[pp + 17]
    A  = p[pp + 23]; B  = p[pp + 24]; kfex = p[pp + 25]

    efd = z[dp + 3]
    vre = z[vri]; vim = z[vri + 1]
    vt = sqrt(vre*vre + vim*vim)
    vt = vt == 0.0 ? 1e-12 : vt

    kf = KF / TF
    r  = TC / TB
    g  = kfex * (KE + 2.0 * B * max(kfex * efd - A, 0.0)) + KD   # dVFE/de_fd
    du_defd = -kf * g
    tek = TE * kfex

    nz[jac_pos[k, J_A2_R1_vc]]  = -1.0 / TR
    nz[jac_pos[k, J_A2_R1_vr]]  = (vre / vt) / TR
    nz[jac_pos[k, J_A2_R1_vi]]  = (vim / vt) / TR

    nz[jac_pos[k, J_A2_R2_vc]]  = -1.0 / TB
    nz[jac_pos[k, J_A2_R2_xf]]  = -1.0 / TB
    nz[jac_pos[k, J_A2_R2_efd]] = du_defd / TB
    nz[jac_pos[k, J_A2_R2_xll]] = -1.0 / TB

    nz[jac_pos[k, J_A2_R3_xll]] = KA * (1.0 - r) / TA
    nz[jac_pos[k, J_A2_R3_vc]]  = -KA * r / TA
    nz[jac_pos[k, J_A2_R3_xf]]  = -KA * r / TA
    nz[jac_pos[k, J_A2_R3_efd]] = KA * r * du_defd / TA
    nz[jac_pos[k, J_A2_R3_va]]  = -1.0 / TA

    if vsi > 0
        nz[jac_pos[k, J_A2_R2_vs]] = 1.0 / TB
        nz[jac_pos[k, J_A2_R3_vs]] = KA * r / TA
    end

    nz[jac_pos[k, J_A2_R4_va]]  = KB / tek
    nz[jac_pos[k, J_A2_R4_efd]] = -(KB * KH + 1.0) * g / tek

    nz[jac_pos[k, J_A2_R5_efd]] = -kf * g / TF
    nz[jac_pos[k, J_A2_R5_xf]]  = -1.0 / TF
    end
    return nothing
end

exac2_preallocate!(coord_list, t::StdExcTable{EXAC2}) =
    _std_exc_preallocate!(_exac2_jac_coords, coord_list, t)
exac2_jac_positions!(t::StdExcTable{EXAC2}, J::SparseMatrixCSC) =
    _std_exc_jac_positions!(_exac2_jac_coords, t, J)
exac2_residual_batch!(f, z, p, t::StdExcTable{EXAC2}) =
    _std_exc_residual_batch!(_exac2_residual_one!, f, z, p, t)
exac2_jacobian_batch!(J::SparseMatrixCSC, z, p, t::StdExcTable{EXAC2}) =
    _std_exc_jacobian_batch!(_exac2_jacobian_one!, J, z, p, t)

attaches_to(::Type{EXAC2}) = Genrou
produces_signals(::Type{EXAC2}) = (
    (target_ctrl_offset = 0,          # Genrou ctrl[0] = e_fd
     source_kind        = :diff_at,
     source_offset      = 3),         # EXAC2 diff[3] = e_fd
)
consumes_signals(::Type{EXAC2}) = ()

# EXAC1 / ESAC1A: same leaves, same pvec layout (KB = 1, KH = 0 written by
# fill_pvec!; see AbstractACExciter in src/exciters_psse.jl).
for (M, pre) in ((:EXAC1, :exac1), (:ESAC1A, :esac1a))
    @eval begin
        $(Symbol(pre, :_preallocate!))(coord_list, t::StdExcTable{$M}) =
            _std_exc_preallocate!(_exac2_jac_coords, coord_list, t)
        $(Symbol(pre, :_jac_positions!))(t::StdExcTable{$M}, J::SparseMatrixCSC) =
            _std_exc_jac_positions!(_exac2_jac_coords, t, J)
        $(Symbol(pre, :_residual_batch!))(f, z, p, t::StdExcTable{$M}) =
            _std_exc_residual_batch!(_exac2_residual_one!, f, z, p, t)
        $(Symbol(pre, :_jacobian_batch!))(J::SparseMatrixCSC, z, p, t::StdExcTable{$M}) =
            _std_exc_jacobian_batch!(_exac2_jacobian_one!, J, z, p, t)
        attaches_to(::Type{$M}) = Genrou
        produces_signals(::Type{$M}) = (
            (target_ctrl_offset = 0, source_kind = :diff_at, source_offset = 3),
        )
        consumes_signals(::Type{$M}) = ()
    end
end
