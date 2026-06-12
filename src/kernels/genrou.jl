# GENROU batched residual & Jacobian.
#
# This file establishes the pattern that IEESGO / ESDC1A / ZIPLoad kernels
# follow:
#   * `genrou_residual_batch!(f, z, u, table)` — flat `@inbounds for i in 1:n`
#     over a `GenrouTable` (SoA), reading parameters from SoA columns and
#     writing scattered global rows of `f` / network voltage rows.
#   * `genrou_jacobian_batch!(J, z, u, table)` — same loop shape, writes
#     directly to `J.nzval[table.jac_pos[i, k]]` (no CSR row search).
#   * `genrou_jac_positions!(table, J, ps)` — runs ONCE at end of
#     `preallocate_jacobian` to fill `table.jac_pos` by looking up each
#     (row, col) the Jacobian kernel will touch.
#
# Math is the verbatim translation of `src/generators.jl::rhs_fun!(::Genrou)`
# / `cinject!(::Genrou)` / `rhs_jac!(::Genrou)`. The parity test in
# `test/test_parity.jl` asserts max|Δ| < 1e-14 vs the legacy methods.

# --------------------------------------------------------------------
# RESIDUAL BATCH
# --------------------------------------------------------------------

"""
    genrou_residual_batch!(f, z, u, table::GenrouTable)

For every GENROU device, write into `f`:
  - 6 diff rows at `diff_ptr+(0..5)`
  - 4 alg rows at `diff_dim + alg_ptr + (0..3)` (caller must pass the
    pre-shifted absolute row index; here we assume `f` is the GLOBAL f
    and the table's `alg_ptr` is the per-block offset — same convention
    as the legacy heterogeneous loop in `src/dynamics.jl::rhs_fun!`)
  - 2 network voltage rows at `net_ptr + 2*(bus-1) + (0..1)` (accumulated
    via `+=`, since multiple devices may share a bus)

The caller is responsible for zeroing `f`'s network block via
`mul!(net_block, ybus_real, v, -1.0, 0.0)` BEFORE calling this kernel;
all generator current-injection contributions are accumulated on top.

Reads `e_fd` from `u[ctrl_ptr]` and `p_m` from `u[ctrl_ptr+1]` (Genrou's
two ctrl slots). When `has_exc[k] == false`, `e_fd` reads 0 from the
zero-initialized control vector; same for `has_gov[k]` and `p_m`.

`diff_dim` and `net_ptr` are passed in to keep the kernel cheap and
independent of `PowerSystem`. Caller:
    diff_dim = sys.dynamic.diff_dim
    alg_dim  = sys.dynamic.alg_dim
    net_ptr  = diff_dim + alg_dim
"""
@inline function genrou_residual_batch!(
        f::AbstractArray, z::AbstractArray, u::AbstractArray, p::AbstractArray,
        table::GenrouTable, diff_dim::Int, net_ptr::Int,
)
    n = table.n
    n == 0 && return nothing
    twopi60 = 2.0 * π * 60.0

    @inbounds for k in 1:n
        # ----- pointers -----
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k])
        cp = Int(table.ctrl_ptr[k])
        pp = Int(table.par_ptr[k])
        bus = Int(table.bus[k])

        # ----- parameters (from pvec; single source of truth for AD path) -----
        x_d    = p[pp]
        x_q    = p[pp + 1]
        x_dp   = p[pp + 2]
        x_qp   = p[pp + 3]
        x_ddp  = p[pp + 4]
        xl     = p[pp + 5]
        H      = p[pp + 6]
        D      = p[pp + 7]
        T_d0p  = p[pp + 8]
        T_q0p  = p[pp + 9]
        T_d0dp = p[pp + 10]
        T_q0dp = p[pp + 11]
        S1     = p[pp + 12]
        S2     = p[pp + 13]
        x_qdp  = x_ddp

        # ----- states (global z vector) -----
        e_qp   = z[dp]
        e_dp   = z[dp + 1]
        phi_1d = z[dp + 2]
        phi_2q = z[dp + 3]
        w      = z[dp + 4]
        delta  = z[dp + 5]
        v_q    = z[diff_dim + ap]
        v_d    = z[diff_dim + ap + 1]
        i_q    = z[diff_dim + ap + 2]
        i_d    = z[diff_dim + ap + 3]

        # ----- voltages -----
        vr_idx = net_ptr + 2*(bus - 1) + 1
        vi_idx = vr_idx + 1
        vr = z[vr_idx]
        vi = z[vi_idx]

        # ----- controls (zero if not wired) -----
        e_fd = u[cp]
        p_m  = u[cp + 1]

        # ----- auxiliary -----
        # Swing eq: f5 = (Pm - D*w - psi_de*iq + psi_qe*id)/(2H).
        # No 1/(1+w) factor on tmech — historical PSS/E form omitted.
        psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp +
                 (x_dp  - x_ddp)/(x_dp - xl)*phi_1d
        psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp +
                  (x_qp  - x_ddp)/(x_qp - xl)*phi_2q

        # Quadratic open-circuit saturation: adds -Se*psi_de to the e_qp eq.
        sat_a, sat_b = _genrou_sat_coefficients(S1, S2)
        psi2 = sqrt(psi_de*psi_de + psi_qe*psi_qe)
        Se = _genrou_sat_se(psi2, sat_a, sat_b)

        # ----- diff residuals -----
        f[dp]     = (-e_qp + e_fd - (i_d - (-x_ddp + x_dp)*(-e_qp + i_d*(x_dp - xl) + phi_1d)/((x_dp - xl)^2)) * (x_d - x_dp) - Se*psi_de) / T_d0p
        f[dp + 1] = (-e_dp +        (i_q - (-x_qdp + x_qp)*( e_dp + i_q*(x_qp - xl) + phi_2q)/((x_qp - xl)^2)) * (x_q - x_qp)) / T_q0p
        f[dp + 2] = ( e_qp - i_d*(x_dp - xl) - phi_1d) / T_d0dp
        f[dp + 3] = (-e_dp - i_q*(x_qp - xl) - phi_2q) / T_q0dp
        f[dp + 4] = (p_m - D*w - psi_de*i_q + psi_qe*i_d) / (2.0 * H)
        f[dp + 5] = twopi60 * w

        # ----- alg residuals (stator currents + park projection) -----
        f[diff_dim + ap]     = i_d - ((x_ddp - xl)/(x_dp - xl)*e_qp +
                                       (x_dp - x_ddp)/(x_dp - xl)*phi_1d - v_q) / x_ddp
        f[diff_dim + ap + 1] = i_q - (-(x_qdp - xl)/(x_qp - xl)*e_dp +
                                       (x_qp - x_qdp)/(x_qp - xl)*phi_2q + v_d) / x_qdp
        sd, cd = sincos(delta)
        f[diff_dim + ap + 2] = v_d - (vr*sd - vi*cd)
        f[diff_dim + ap + 3] = v_q - (vr*cd + vi*sd)

        # ----- network current injection (accumulate, NOT assign) -----
        f[vr_idx] += sd*i_d + cd*i_q
        f[vi_idx] += -cd*i_d + sd*i_q
    end
    return nothing
end

# --------------------------------------------------------------------
# JACOBIAN BATCH — direct nzval writes via precomputed positions
# --------------------------------------------------------------------
#
# Slot layout for `table.jac_pos[k, slot]` (one row of the SoA matrix
# carries the J.nzval indices for one device's Jacobian entries).
# MUST agree with `genrou_jac_positions!` AND `genrou_jacobian_batch!`.

# Diff row 1 (df1/d{e_qp, phi_1d, i_d, e_dp, phi_2q})
# e_dp and phi_2q columns are present whenever saturation is on (S2>0); the
# slot is allocated always, written to 0 when saturation is off.
const J_GR_R1_eqp   = 1
const J_GR_R1_phi1d = 2
const J_GR_R1_id    = 3
const J_GR_R1_edp   = 44
const J_GR_R1_phi2q = 45
# Diff row 2 (df2/d{e_dp, phi_2q, i_q})
const J_GR_R2_edp   = 4
const J_GR_R2_phi2q = 5
const J_GR_R2_iq    = 6
# Diff row 3 (df3/d{e_qp, phi_1d, i_d})
const J_GR_R3_eqp   = 7
const J_GR_R3_phi1d = 8
const J_GR_R3_id    = 9
# Diff row 4 (df4/d{e_dp, phi_2q, i_q})
const J_GR_R4_edp   = 10
const J_GR_R4_phi2q = 11
const J_GR_R4_iq    = 12
# Diff row 5 (df5/d{e_qp, e_dp, phi_1d, phi_2q, w, i_q, i_d})
const J_GR_R5_eqp   = 13
const J_GR_R5_edp   = 14
const J_GR_R5_phi1d = 15
const J_GR_R5_phi2q = 16
const J_GR_R5_w     = 17
const J_GR_R5_iq    = 18
const J_GR_R5_id    = 19
# Diff row 6 (df6/dw)
const J_GR_R6_w     = 20
# Alg row 1 (df_alg1/d{e_qp, phi_1d, v_q, i_d})
const J_GR_A1_eqp   = 21
const J_GR_A1_phi1d = 22
const J_GR_A1_vq    = 23
const J_GR_A1_id    = 24
# Alg row 2 (df_alg2/d{e_dp, phi_2q, v_d, i_q})
const J_GR_A2_edp   = 25
const J_GR_A2_phi2q = 26
const J_GR_A2_vd    = 27
const J_GR_A2_iq    = 28
# Alg row 3 (df_alg3/d{delta, v_d, vr, vi})
const J_GR_A3_delta = 29
const J_GR_A3_vd    = 30
const J_GR_A3_vr    = 31
const J_GR_A3_vi    = 32
# Alg row 4 (df_alg4/d{delta, v_q, vr, vi})
const J_GR_A4_delta = 33
const J_GR_A4_vq    = 34
const J_GR_A4_vr    = 35
const J_GR_A4_vi    = 36
# Network row vr (dfvr/d{delta, i_q, i_d})
const J_GR_NR_delta = 37
const J_GR_NR_iq    = 38
const J_GR_NR_id    = 39
# Network row vi (dfvi/d{delta, i_q, i_d})
const J_GR_NI_delta = 40
const J_GR_NI_iq    = 41
const J_GR_NI_id    = 42
# Optional cross-coupling: diff row 5 ∂/∂p_m (when GENROU is wired to a governor).
# When has_gov[k] is false the slot stays 0 in jac_pos and the kernel skips it.
const J_GR_R5_pm    = 43
# Optional cross-coupling: diff row 1 ∂/∂e_fd (when GENROU is wired to an exciter).
const J_GR_R1_efd   = 46

# Sanity check: must match GENROU_JAC_NENTRIES declared in tables/genrou.jl.
@assert J_GR_R1_efd == GENROU_JAC_NENTRIES "Slot table out of sync with GENROU_JAC_NENTRIES"

"""
    genrou_coupling_preallocate!(coord_list, table::GenrouTable)

For every GENROU wired to a governor, push the cross-coupling sparsity
entry `(dp+4, pm_idx[k])` — i.e. ∂(swing eq)/∂p_m — into coord_list.
The legacy per-device GENROU preallocator hardcodes `governor = false`
and cannot see the layout's wiring; this pass adds the missing entry.
"""
function genrou_coupling_preallocate!(coord_list::Vector{Vector{Int}},
                                       table::GenrouTable)
    for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        if table.has_gov[k]
            push!(coord_list[dp + 4], Int(table.pm_idx[k]))
        end
        if table.has_exc[k]
            push!(coord_list[dp], Int(table.efd_idx[k]))
        end
    end
    return nothing
end

"""
    genrou_jac_positions!(table::GenrouTable, J::SparseMatrixCSC,
                          diff_dim::Int, net_ptr::Int)

For every GENROU device, look up the index into `J.nzval` for each of
the 42 (row, col) entries the Jacobian kernel writes, and store it in
`table.jac_pos[k, slot]`. Must be called once after `preallocate_jacobian`
returns, before any `genrou_jacobian_batch!` call.

The lookup is a single linear scan within `nzrange(J, col)` per entry —
~O(log(column density)) amortized. Cost paid once; thereafter each
Newton iteration's Jacobian assembly is one direct write per entry.
"""
function genrou_jac_positions!(
        table::GenrouTable, J::SparseMatrixCSC,
        diff_dim::Int, net_ptr::Int,
)
    n = table.n
    n == 0 && return nothing

    rows = rowvals(J)
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k])
        bus = Int(table.bus[k])

        e_qp_idx   = dp
        e_dp_idx   = dp + 1
        phi_1d_idx = dp + 2
        phi_2q_idx = dp + 3
        w_idx      = dp + 4
        delta_idx  = dp + 5
        v_q_idx    = diff_dim + ap
        v_d_idx    = diff_dim + ap + 1
        i_q_idx    = diff_dim + ap + 2
        i_d_idx    = diff_dim + ap + 3
        vr_idx     = net_ptr + 2*(bus - 1) + 1
        vi_idx     = vr_idx + 1

        # _find_pos(J, rows, row, col) — row = equation index, col = state variable index.
        # i.e. entry J[row, col] = ∂f[row]/∂z[col].
        # Diff row 1
        table.jac_pos[k, J_GR_R1_eqp]   = _find_pos(J, rows, dp,     e_qp_idx)
        table.jac_pos[k, J_GR_R1_phi1d] = _find_pos(J, rows, dp,     phi_1d_idx)
        table.jac_pos[k, J_GR_R1_id]    = _find_pos(J, rows, dp,     i_d_idx)
        table.jac_pos[k, J_GR_R1_edp]   = _find_pos(J, rows, dp,     e_dp_idx)
        table.jac_pos[k, J_GR_R1_phi2q] = _find_pos(J, rows, dp,     phi_2q_idx)
        # Diff row 2
        table.jac_pos[k, J_GR_R2_edp]   = _find_pos(J, rows, dp + 1, e_dp_idx)
        table.jac_pos[k, J_GR_R2_phi2q] = _find_pos(J, rows, dp + 1, phi_2q_idx)
        table.jac_pos[k, J_GR_R2_iq]    = _find_pos(J, rows, dp + 1, i_q_idx)
        # Diff row 3
        table.jac_pos[k, J_GR_R3_eqp]   = _find_pos(J, rows, dp + 2, e_qp_idx)
        table.jac_pos[k, J_GR_R3_phi1d] = _find_pos(J, rows, dp + 2, phi_1d_idx)
        table.jac_pos[k, J_GR_R3_id]    = _find_pos(J, rows, dp + 2, i_d_idx)
        # Diff row 4
        table.jac_pos[k, J_GR_R4_edp]   = _find_pos(J, rows, dp + 3, e_dp_idx)
        table.jac_pos[k, J_GR_R4_phi2q] = _find_pos(J, rows, dp + 3, phi_2q_idx)
        table.jac_pos[k, J_GR_R4_iq]    = _find_pos(J, rows, dp + 3, i_q_idx)
        # Diff row 5
        table.jac_pos[k, J_GR_R5_eqp]   = _find_pos(J, rows, dp + 4, e_qp_idx)
        table.jac_pos[k, J_GR_R5_edp]   = _find_pos(J, rows, dp + 4, e_dp_idx)
        table.jac_pos[k, J_GR_R5_phi1d] = _find_pos(J, rows, dp + 4, phi_1d_idx)
        table.jac_pos[k, J_GR_R5_phi2q] = _find_pos(J, rows, dp + 4, phi_2q_idx)
        table.jac_pos[k, J_GR_R5_w]     = _find_pos(J, rows, dp + 4, w_idx)
        table.jac_pos[k, J_GR_R5_iq]    = _find_pos(J, rows, dp + 4, i_q_idx)
        table.jac_pos[k, J_GR_R5_id]    = _find_pos(J, rows, dp + 4, i_d_idx)
        # Diff row 6
        table.jac_pos[k, J_GR_R6_w]     = _find_pos(J, rows, dp + 5, w_idx)
        # Alg row 1 (global row diff_dim + ap)
        table.jac_pos[k, J_GR_A1_eqp]   = _find_pos(J, rows, diff_dim + ap,     e_qp_idx)
        table.jac_pos[k, J_GR_A1_phi1d] = _find_pos(J, rows, diff_dim + ap,     phi_1d_idx)
        table.jac_pos[k, J_GR_A1_vq]    = _find_pos(J, rows, diff_dim + ap,     v_q_idx)
        table.jac_pos[k, J_GR_A1_id]    = _find_pos(J, rows, diff_dim + ap,     i_d_idx)
        # Alg row 2
        table.jac_pos[k, J_GR_A2_edp]   = _find_pos(J, rows, diff_dim + ap + 1, e_dp_idx)
        table.jac_pos[k, J_GR_A2_phi2q] = _find_pos(J, rows, diff_dim + ap + 1, phi_2q_idx)
        table.jac_pos[k, J_GR_A2_vd]    = _find_pos(J, rows, diff_dim + ap + 1, v_d_idx)
        table.jac_pos[k, J_GR_A2_iq]    = _find_pos(J, rows, diff_dim + ap + 1, i_q_idx)
        # Alg row 3
        table.jac_pos[k, J_GR_A3_delta] = _find_pos(J, rows, diff_dim + ap + 2, delta_idx)
        table.jac_pos[k, J_GR_A3_vd]    = _find_pos(J, rows, diff_dim + ap + 2, v_d_idx)
        table.jac_pos[k, J_GR_A3_vr]    = _find_pos(J, rows, diff_dim + ap + 2, vr_idx)
        table.jac_pos[k, J_GR_A3_vi]    = _find_pos(J, rows, diff_dim + ap + 2, vi_idx)
        # Alg row 4
        table.jac_pos[k, J_GR_A4_delta] = _find_pos(J, rows, diff_dim + ap + 3, delta_idx)
        table.jac_pos[k, J_GR_A4_vq]    = _find_pos(J, rows, diff_dim + ap + 3, v_q_idx)
        table.jac_pos[k, J_GR_A4_vr]    = _find_pos(J, rows, diff_dim + ap + 3, vr_idx)
        table.jac_pos[k, J_GR_A4_vi]    = _find_pos(J, rows, diff_dim + ap + 3, vi_idx)
        # Network row vr
        table.jac_pos[k, J_GR_NR_delta] = _find_pos(J, rows, vr_idx, delta_idx)
        table.jac_pos[k, J_GR_NR_iq]    = _find_pos(J, rows, vr_idx, i_q_idx)
        table.jac_pos[k, J_GR_NR_id]    = _find_pos(J, rows, vr_idx, i_d_idx)
        # Network row vi
        table.jac_pos[k, J_GR_NI_delta] = _find_pos(J, rows, vi_idx, delta_idx)
        table.jac_pos[k, J_GR_NI_iq]    = _find_pos(J, rows, vi_idx, i_q_idx)
        table.jac_pos[k, J_GR_NI_id]    = _find_pos(J, rows, vi_idx, i_d_idx)
        # Optional cross-coupling: ∂f5/∂p_m (only when wired to a governor).
        # Slot stays 0 (sentinel) when has_gov[k] is false; kernel skips it.
        if table.has_gov[k]
            table.jac_pos[k, J_GR_R5_pm] = _find_pos(J, rows, dp + 4, Int(table.pm_idx[k]))
        end
        # Optional cross-coupling: ∂f1/∂e_fd (only when wired to an exciter).
        if table.has_exc[k]
            table.jac_pos[k, J_GR_R1_efd] = _find_pos(J, rows, dp, Int(table.efd_idx[k]))
        end
    end

    # Sanity gate: every always-present slot must be populated. Optional
    # cross-coupling columns are allowed to be 0 when their wire is absent.
    @inbounds for k in 1:n
        for slot in 1:GENROU_JAC_NENTRIES
            slot == J_GR_R5_pm && continue
            slot == J_GR_R1_efd && continue
            @assert table.jac_pos[k, slot] != 0 "GENROU jac_pos[$k, $slot] is zero — sparsity pattern mismatch"
        end
        if table.has_gov[k]
            @assert table.jac_pos[k, J_GR_R5_pm] != 0 "GENROU jac_pos[$k, J_GR_R5_pm] is zero but governor is wired"
        end
        if table.has_exc[k]
            @assert table.jac_pos[k, J_GR_R1_efd] != 0 "GENROU jac_pos[$k, J_GR_R1_efd] is zero but exciter is wired"
        end
    end
    return nothing
end

# Look up the index in `J.nzval` for entry `(row, col)`. Throws if not
# present in the preallocated sparsity pattern — that's a bug, not a
# runtime fallback case.
@inline function _find_pos(J::SparseMatrixCSC, rows::AbstractVector, row::Integer, col::Integer)
    for idx in nzrange(J, col)
        rows[idx] == row && return Int32(idx)
    end
    error("(row=$row, col=$col) not in J's sparsity pattern — genrou_jac_positions! / preallocate_jacobian! mismatch")
end

"""
    genrou_jacobian_batch!(J, z, u, table::GenrouTable, diff_dim, net_ptr)

For every GENROU device, write the 42 Jacobian entries directly to
`J.nzval[table.jac_pos[k, slot]]`. No CSR row search per iteration.

The current-injection rows (`vr`, `vi`) are ACCUMULATED via `+=` because
multiple devices can share a bus and the kernel for each device adds its
contribution on top of whatever's already there (typically the network
admittance, written by `current_injection_jacobian!` first).

All other rows are assigned (`=`) because they're 1:1 with a single
device and the cumulative Jacobian-zero fill in `beuler_jac!` already
cleared them.
"""
@inline function genrou_jacobian_batch!(
        J::SparseMatrixCSC, z::AbstractArray, u::AbstractArray, p::AbstractArray,
        table::GenrouTable, diff_dim::Int, net_ptr::Int,
)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    twopi60 = 2.0 * π * 60.0

    @inbounds for k in 1:n
        # ----- pointers -----
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k])
        pp = Int(table.par_ptr[k])
        bus = Int(table.bus[k])

        # ----- parameters (from pvec) -----
        x_d    = p[pp]
        x_q    = p[pp + 1]
        x_dp   = p[pp + 2]
        x_qp   = p[pp + 3]
        x_ddp  = p[pp + 4]
        xl     = p[pp + 5]
        H      = p[pp + 6]
        D      = p[pp + 7]
        T_d0p  = p[pp + 8]
        T_q0p  = p[pp + 9]
        T_d0dp = p[pp + 10]
        T_q0dp = p[pp + 11]
        S1     = p[pp + 12]
        S2     = p[pp + 13]
        x_qdp  = x_ddp

        # ----- states -----
        e_qp   = z[dp]
        e_dp   = z[dp + 1]
        phi_1d = z[dp + 2]
        phi_2q = z[dp + 3]
        w      = z[dp + 4]
        delta  = z[dp + 5]
        i_q    = z[diff_dim + ap + 2]
        i_d    = z[diff_dim + ap + 3]

        sd, cd = sincos(delta)

        # ----- saturation contributions to row dp (e_qp eq) -----
        # Slots J_GR_R1_edp / J_GR_R1_phi2q are always allocated; when
        # saturation is off (S2≤0 or psi2 ≤ sat_a) Se=dSe_dpsi=0 and the
        # increments collapse to 0.
        psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp +
                 (x_dp  - x_ddp)/(x_dp - xl)*phi_1d
        psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp +
                  (x_qp  - x_ddp)/(x_qp - xl)*phi_2q
        sat_a, sat_b = _genrou_sat_coefficients(S1, S2)
        psi2 = sqrt(psi_de*psi_de + psi_qe*psi_qe)
        Se = _genrou_sat_se(psi2, sat_a, sat_b)
        if sat_b == 0.0 || psi2 <= sat_a || psi2 == 0.0
            dSe_dpsi = 0.0
        else
            g = psi2 - sat_a
            dSe_dpsi = sat_b * (2.0 * g * psi2 - g * g) / (psi2 * psi2)
        end
        dpsi_dpsi_de = psi2 == 0.0 ? 0.0 : psi_de / psi2
        dpsi_dpsi_qe = psi2 == 0.0 ? 0.0 : psi_qe / psi2
        dpsi_de_deqp   = (x_ddp - xl) / (x_dp - xl)
        dpsi_de_phi1d  = (x_dp - x_ddp) / (x_dp - xl)
        dpsi_qe_dedp   = -(x_ddp - xl) / (x_qp - xl)
        dpsi_qe_phi2q  = (x_qp - x_ddp) / (x_qp - xl)
        dSe_dpsi_de = dSe_dpsi * dpsi_dpsi_de
        dSe_dpsi_qe = dSe_dpsi * dpsi_dpsi_qe
        # d(-Se*psi_de)/d*: chain rule via psi_de and psi_qe (both feed psi2 → Se).
        dT_dpsi_de = -(dSe_dpsi_de * psi_de + Se)
        dT_dpsi_qe = -(dSe_dpsi_qe * psi_de)
        dT_deqp_sat   = dT_dpsi_de * dpsi_de_deqp / T_d0p
        dT_dphi1d_sat = dT_dpsi_de * dpsi_de_phi1d / T_d0p
        dT_dedp_sat   = dT_dpsi_qe * dpsi_qe_dedp / T_d0p
        dT_dphi2q_sat = dT_dpsi_qe * dpsi_qe_phi2q / T_d0p

        # ===== diff row 1 =====
        nz[table.jac_pos[k, J_GR_R1_eqp]]   = (-(x_d - x_dp)*(-x_ddp + x_dp)*(x_dp - xl)^(-2.0) - 1) / T_d0p + dT_deqp_sat
        nz[table.jac_pos[k, J_GR_R1_phi1d]] =  (x_d - x_dp)*(-x_ddp + x_dp)*(x_dp - xl)^(-2.0) / T_d0p + dT_dphi1d_sat
        nz[table.jac_pos[k, J_GR_R1_id]]    = -(x_d - x_dp)*(-(-x_ddp + x_dp)*(x_dp - xl)^(-1.0) + 1) / T_d0p
        nz[table.jac_pos[k, J_GR_R1_edp]]   = dT_dedp_sat
        nz[table.jac_pos[k, J_GR_R1_phi2q]] = dT_dphi2q_sat
        # Optional cross-coupling ∂f1/∂e_fd = 1/T_d0p (only when wired to an exciter).
        if table.has_exc[k]
            nz[table.jac_pos[k, J_GR_R1_efd]] = 1.0 / T_d0p
        end

        # ===== diff row 2 =====
        nz[table.jac_pos[k, J_GR_R2_edp]]   = (-(x_q - x_qp)*(-x_qdp + x_qp)*(x_qp - xl)^(-2.0) - 1) / T_q0p
        nz[table.jac_pos[k, J_GR_R2_phi2q]] = -(x_q - x_qp)*(-x_qdp + x_qp)*(x_qp - xl)^(-2.0) / T_q0p
        nz[table.jac_pos[k, J_GR_R2_iq]]    =  (x_q - x_qp)*(-(-x_qdp + x_qp)*(x_qp - xl)^(-1.0) + 1) / T_q0p

        # ===== diff row 3 =====
        nz[table.jac_pos[k, J_GR_R3_eqp]]   =  1.0 / T_d0dp
        nz[table.jac_pos[k, J_GR_R3_phi1d]] = -1.0 / T_d0dp
        nz[table.jac_pos[k, J_GR_R3_id]]    = (-x_dp + xl) / T_d0dp

        # ===== diff row 4 =====
        nz[table.jac_pos[k, J_GR_R4_edp]]   = -1.0 / T_q0dp
        nz[table.jac_pos[k, J_GR_R4_phi2q]] = -1.0 / T_q0dp
        nz[table.jac_pos[k, J_GR_R4_iq]]    = (-x_qp + xl) / T_q0dp

        # ===== diff row 5 =====
        nz[table.jac_pos[k, J_GR_R5_eqp]]   = -0.5 * i_q * (x_ddp - xl) / (H * (x_dp - xl))
        nz[table.jac_pos[k, J_GR_R5_edp]]   =  0.5 * i_d * (-x_ddp + xl) / (H * (x_qp - xl))
        nz[table.jac_pos[k, J_GR_R5_phi1d]] = -0.5 * i_q * (-x_ddp + x_dp) / (H * (x_dp - xl))
        nz[table.jac_pos[k, J_GR_R5_phi2q]] =  0.5 * i_d * (-x_ddp + x_qp) / (H * (x_qp - xl))
        nz[table.jac_pos[k, J_GR_R5_w]]     = -0.5 * D / H
        # Optional cross-coupling ∂f5/∂p_m = 1/(2H), only when wired.
        if table.has_gov[k]
            nz[table.jac_pos[k, J_GR_R5_pm]] = 0.5 / H
        end
        nz[table.jac_pos[k, J_GR_R5_iq]]    =  0.5 * (-e_qp * (x_ddp - xl) / (x_dp - xl) - phi_1d * (-x_ddp + x_dp) / (x_dp - xl)) / H
        nz[table.jac_pos[k, J_GR_R5_id]]    =  0.5 * ( e_dp * (-x_ddp + xl) / (x_qp - xl) + phi_2q * (-x_ddp + x_qp) / (x_qp - xl)) / H

        # ===== diff row 6 =====
        nz[table.jac_pos[k, J_GR_R6_w]]     = twopi60

        # ===== alg row 1 =====
        nz[table.jac_pos[k, J_GR_A1_eqp]]   = -(x_ddp - xl) / (x_ddp * (x_dp - xl))
        nz[table.jac_pos[k, J_GR_A1_phi1d]] = -(-x_ddp + x_dp) / (x_ddp * (x_dp - xl))
        nz[table.jac_pos[k, J_GR_A1_vq]]    =  1.0 / x_ddp
        nz[table.jac_pos[k, J_GR_A1_id]]    =  1.0

        # ===== alg row 2 =====
        nz[table.jac_pos[k, J_GR_A2_edp]]   = -(-x_qdp + xl) / (x_qdp * (x_qp - xl))
        nz[table.jac_pos[k, J_GR_A2_phi2q]] = -(-x_qdp + x_qp) / (x_qdp * (x_qp - xl))
        nz[table.jac_pos[k, J_GR_A2_vd]]    = -1.0 / x_qdp
        nz[table.jac_pos[k, J_GR_A2_iq]]    =  1.0

        # ===== alg row 3 =====
        vr_idx = net_ptr + 2*(bus - 1) + 1
        vi_idx = vr_idx + 1
        vr = z[vr_idx]
        vi = z[vi_idx]
        nz[table.jac_pos[k, J_GR_A3_delta]] = -vr*cd - vi*sd
        nz[table.jac_pos[k, J_GR_A3_vd]]    =  1.0
        nz[table.jac_pos[k, J_GR_A3_vr]]    = -sd
        nz[table.jac_pos[k, J_GR_A3_vi]]    =  cd

        # ===== alg row 4 =====
        nz[table.jac_pos[k, J_GR_A4_delta]] = vr*sd - vi*cd
        nz[table.jac_pos[k, J_GR_A4_vq]]    = 1.0
        nz[table.jac_pos[k, J_GR_A4_vr]]    = -cd
        nz[table.jac_pos[k, J_GR_A4_vi]]    = -sd

        # ===== network rows (current injection, ACCUMULATE via +=) =====
        # vr-row entries
        nz[table.jac_pos[k, J_GR_NR_delta]] += i_d*cd - i_q*sd
        nz[table.jac_pos[k, J_GR_NR_iq]]    += cd
        nz[table.jac_pos[k, J_GR_NR_id]]    += sd
        # vi-row entries
        nz[table.jac_pos[k, J_GR_NI_delta]] += i_d*sd + i_q*cd
        nz[table.jac_pos[k, J_GR_NI_iq]]    += sd
        nz[table.jac_pos[k, J_GR_NI_id]]    += -cd
    end
    return nothing
end
