# ESAC6A batched residual & Jacobian leaves. Model, simplifications and
# parameter layout: see the ESAC6A header in src/exciters_psse.jl.
#
# States (dp+0..3): vc, xk, xll, e_fd.  pvec (pp+k is slot k+1):
#   TR=0 KA=1 TA=2 TK=3 TB=4 TC=5 TE=10 KD=17 KE=18 A=23 B=24 kfex=25 vref=26
#
#   u   = vref − vc + vs
#   y   = xk + (TK/TA)·(KA·u − xk) ;  VR = xll + (TC/TB)·(y − xll)
#   VFE = KE·kfex·e_fd + B·max(kfex·e_fd − A, 0)² + KD·e_fd
#   F[dp+0] = (vt − vc)/TR
#   F[dp+1] = (KA·u − xk)/TA
#   F[dp+2] = (y − xll)/TB
#   F[dp+3] = (VR − VFE)/(TE·kfex)

const ESAC6A_JAC_NENTRIES = 15

const J_A6_R1_vc  = 1;  const J_A6_R1_vr  = 2;  const J_A6_R1_vi  = 3
const J_A6_R2_vc  = 4;  const J_A6_R2_xk  = 5;  const J_A6_R2_vs  = 6
const J_A6_R3_xk  = 7;  const J_A6_R3_vc  = 8;  const J_A6_R3_xll = 9;  const J_A6_R3_vs = 10
const J_A6_R4_xll = 11; const J_A6_R4_xk  = 12; const J_A6_R4_vc  = 13
const J_A6_R4_efd = 14; const J_A6_R4_vs  = 15

# State offsets: vc=0 xk=1 xll=2 efd=3
@inline _esac6a_jac_coords(dp::Int, vr::Int, vsi::Int) = (
    (dp,     dp),     (dp,     vr),     (dp,     vr + 1),
    (dp + 1, dp),     (dp + 1, dp + 1), (dp + 1, vsi),
    (dp + 2, dp + 1), (dp + 2, dp),     (dp + 2, dp + 2), (dp + 2, vsi),
    (dp + 3, dp + 2), (dp + 3, dp + 1), (dp + 3, dp),     (dp + 3, dp + 3), (dp + 3, vsi),
)

@inline function _esac6a_residual_one!(f, z, p,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k::Int)
    @inbounds begin
    dp  = Int(diff_ptr[k])
    pp  = Int(par_ptr[k])
    vri = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

    TR = p[pp];      KA = p[pp + 1]; TA = p[pp + 2]; TK = p[pp + 3]
    TB = p[pp + 4];  TC = p[pp + 5]; TE = p[pp + 10]
    KD = p[pp + 17]; KE = p[pp + 18]
    A  = p[pp + 23]; B  = p[pp + 24]; kfex = p[pp + 25]; vref = p[pp + 26]

    vc = z[dp]; xk = z[dp + 1]; xll = z[dp + 2]; efd = z[dp + 3]
    vre = z[vri]; vim = z[vri + 1]
    vt = sqrt(vre*vre + vim*vim)
    vs = vsi > 0 ? z[vsi] : 0.0

    u    = vref - vc + vs
    y    = xk + (TK / TA) * (KA * u - xk)
    vr   = xll + (TC / TB) * (y - xll)
    ve   = kfex * efd
    dsat = max(ve - A, 0.0)
    vfe  = KE * ve + B * dsat * dsat + KD * efd

    f[dp]     = (vt - vc) / TR
    f[dp + 1] = (KA * u - xk) / TA
    f[dp + 2] = (y - xll) / TB
    f[dp + 3] = (vr - vfe) / (TE * kfex)
    end
    return nothing
end

@inline function _esac6a_jacobian_one!(nz, z, p,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k::Int)
    @inbounds begin
    dp  = Int(diff_ptr[k])
    pp  = Int(par_ptr[k])
    vri = Int(vr_idx_arr[k])
    vsi = Int(vs_idx_arr[k])

    TR = p[pp];      KA = p[pp + 1]; TA = p[pp + 2]; TK = p[pp + 3]
    TB = p[pp + 4];  TC = p[pp + 5]; TE = p[pp + 10]
    KD = p[pp + 17]; KE = p[pp + 18]
    A  = p[pp + 23]; B  = p[pp + 24]; kfex = p[pp + 25]

    efd = z[dp + 3]
    vre = z[vri]; vim = z[vri + 1]
    vt = sqrt(vre*vre + vim*vim)
    vt = vt == 0.0 ? 1e-12 : vt

    r1  = TK / TA
    r2  = TC / TB
    g   = kfex * (KE + 2.0 * B * max(kfex * efd - A, 0.0)) + KD   # dVFE/de_fd
    tek = TE * kfex

    nz[jac_pos[k, J_A6_R1_vc]]  = -1.0 / TR
    nz[jac_pos[k, J_A6_R1_vr]]  = (vre / vt) / TR
    nz[jac_pos[k, J_A6_R1_vi]]  = (vim / vt) / TR

    nz[jac_pos[k, J_A6_R2_vc]]  = -KA / TA
    nz[jac_pos[k, J_A6_R2_xk]]  = -1.0 / TA

    nz[jac_pos[k, J_A6_R3_xk]]  = (1.0 - r1) / TB
    nz[jac_pos[k, J_A6_R3_vc]]  = -r1 * KA / TB
    nz[jac_pos[k, J_A6_R3_xll]] = -1.0 / TB

    nz[jac_pos[k, J_A6_R4_xll]] = (1.0 - r2) / tek
    nz[jac_pos[k, J_A6_R4_xk]]  = r2 * (1.0 - r1) / tek
    nz[jac_pos[k, J_A6_R4_vc]]  = -r2 * r1 * KA / tek
    nz[jac_pos[k, J_A6_R4_efd]] = -g / tek

    if vsi > 0
        nz[jac_pos[k, J_A6_R2_vs]] = KA / TA
        nz[jac_pos[k, J_A6_R3_vs]] = r1 * KA / TB
        nz[jac_pos[k, J_A6_R4_vs]] = r2 * r1 * KA / tek
    end
    end
    return nothing
end

esac6a_preallocate!(coord_list, t::StdExcTable{ESAC6A}) =
    _std_exc_preallocate!(_esac6a_jac_coords, coord_list, t)
esac6a_jac_positions!(t::StdExcTable{ESAC6A}, J::SparseMatrixCSC) =
    _std_exc_jac_positions!(_esac6a_jac_coords, t, J)
esac6a_residual_batch!(f, z, p, t::StdExcTable{ESAC6A}) =
    _std_exc_residual_batch!(_esac6a_residual_one!, f, z, p, t)
esac6a_jacobian_batch!(J::SparseMatrixCSC, z, p, t::StdExcTable{ESAC6A}) =
    _std_exc_jacobian_batch!(_esac6a_jacobian_one!, J, z, p, t)

attaches_to(::Type{ESAC6A}) = Genrou
produces_signals(::Type{ESAC6A}) = (
    (target_ctrl_offset = 0,          # Genrou ctrl[0] = e_fd
     source_kind        = :diff_at,
     source_offset      = 3),         # ESAC6A diff[3] = e_fd
)
consumes_signals(::Type{ESAC6A}) = ()
