# Phase 1 of ROADMAP.md (agent A1.1): builder for GenrouTable.
#
# Phase 2.1: also publishes GENROU_JAC_NENTRIES — the fixed count of
# (row, col) Jacobian entries each GENROU device contributes. The order
# is defined by `genrou_jac_positions!` / `genrou_jacobian_batch!` in
# src/kernels/genrou.jl; both must agree on the slot-to-entry mapping.

# 42 entries: 6 diff rows (3+3+3+3+8+1) + 4 alg rows (4+4+4+4) + 2 net rows (3+3).
# Row tallies (one-based diff row k):
#   diff[1]: 3, diff[2]: 3, diff[3]: 3, diff[4]: 3, diff[5]: 7+(1 if gov), diff[6]: 1
#   With governor wiring the Phase 2.1 batched kernel writes the w_idx coupling
#   directly via the d_pm column entry. Today GENROU has no exciter wiring
#   active in tracked cases; we reserve no slot for e_fd (0 entries) and a
#   single optional slot for p_m → see GENROU_JAC_PM_COL_SLOT below.
const GENROU_JAC_NENTRIES = 45  # 42 base + 1 optional pm-col + 2 saturation cross-cols
const GENROU_JAC_PM_COL_SLOT = 43  # ∂f5/∂p_m when has_gov[k]; 0-fill when not wired

#
# This file is included from src/GradPower.jl AFTER both layout.jl (for the
# GenrouTable type) and dynamics.jl (which transitively includes
# generators.jl for the Genrou struct).
#
# The stub `build_genrou_table` defined in src/layout.jl delegates to
# `_build_genrou_table_impl` below.

"""
    _build_genrou_table_impl(psd) -> GenrouTable

Scan `psd.devices` once and emit a struct-of-arrays table for every Genrou
generator. All vectors have length `n` = number of Genrou devices.

Control coupling fields (`has_gov`, `pm_idx`, `has_exc`, `efd_idx`) are
resolved from `psd.uvec_idx`, which `set_dynamics!` populates BEFORE
`build_layout!` is called.

`jac_pos` is allocated as an `n × 0` Int32 matrix; Phase 2 widens its
second dimension via `preallocate_jacobian`.
"""
function _build_genrou_table_impl(psd)
    # 1. Count generators. Using `isa Genrou` for narrowness — when GENSAL
    #    or another AbstractGeneratorType subtype is added, this filter
    #    will need to be revisited (likely a separate table per subtype).
    n = 0
    for device in psd.devices
        if device.dtype isa Genrou
            n += 1
        end
    end

    # 2. Allocate global pointer vectors.
    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    alg_ptr  = Vector{Int32}(undef, n)
    ctrl_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    # 3. Allocate parameter vectors (14 Genrou parameters incl. S1, S2).
    x_d    = Vector{Float64}(undef, n)
    x_q    = Vector{Float64}(undef, n)
    x_dp   = Vector{Float64}(undef, n)
    x_qp   = Vector{Float64}(undef, n)
    x_ddp  = Vector{Float64}(undef, n)
    xl     = Vector{Float64}(undef, n)
    H      = Vector{Float64}(undef, n)
    D      = Vector{Float64}(undef, n)
    T_d0p  = Vector{Float64}(undef, n)
    T_q0p  = Vector{Float64}(undef, n)
    T_d0dp = Vector{Float64}(undef, n)
    T_q0dp = Vector{Float64}(undef, n)
    S1     = Vector{Float64}(undef, n)
    S2     = Vector{Float64}(undef, n)

    # 4. Allocate control-coupling vectors.
    has_gov = Vector{Bool}(undef, n)
    has_exc = Vector{Bool}(undef, n)
    pm_idx  = Vector{Int32}(undef, n)
    efd_idx = Vector{Int32}(undef, n)

    # 5. Phase 2.1 (A2.0): jac_pos has a fixed 42 entries per Genrou device
    #    (the count of nonzero (row, col) entries the GENROU Jacobian
    #    writes — see src/kernels/genrou.jl GENROU_JAC_NENTRIES).
    #    Allocated zero-filled; populated by `genrou_jac_positions!` when
    #    `preallocate_jacobian` runs. Zero is a "not yet populated"
    #    sentinel — Phase 2.5 batched Jacobian will @assert against it.
    jac_pos = zeros(Int32, n, GENROU_JAC_NENTRIES)

    # 6. Single pass over devices, fill row k for each Genrou match.
    uvec = psd.uvec_idx
    k = 0
    for device in psd.devices
        device.dtype isa Genrou || continue
        k += 1
        gen = device.dtype

        bus[k]      = Int32(gen.bus)
        diff_ptr[k] = Int32(device.diff_ptr)
        alg_ptr[k]  = Int32(device.alg_ptr)
        ctrl_ptr[k] = Int32(device.ctrl_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        # Parameters are already ratio-adjusted by set_ratio! earlier in
        # set_dynamics!, so reading them here is correct.
        x_d[k]    = gen.x_d
        x_q[k]    = gen.x_q
        x_dp[k]   = gen.x_dp
        x_qp[k]   = gen.x_qp
        x_ddp[k]  = gen.x_ddp
        xl[k]     = gen.xl
        H[k]      = gen.H
        D[k]      = gen.D
        T_d0p[k]  = gen.T_d0p
        T_q0p[k]  = gen.T_q0p
        T_d0dp[k] = gen.T_d0dp
        T_q0dp[k] = gen.T_q0dp
        S1[k]     = gen.S1
        S2[k]     = gen.S2

        # Control coupling: ctrl_ptr points at u[1] (e_fd); ctrl_ptr+1 is u[2] (p_m).
        # uvec_idx[slot] == 0 means no controller wired to that slot.
        cp = device.ctrl_ptr
        efd_routed = uvec[cp]
        pm_routed  = uvec[cp + 1]
        has_exc[k] = efd_routed != 0
        efd_idx[k] = Int32(efd_routed)
        has_gov[k] = pm_routed != 0
        pm_idx[k]  = Int32(pm_routed)
    end

    return GenrouTable(n, bus, diff_ptr, alg_ptr, ctrl_ptr, par_ptr,
        x_d, x_q, x_dp, x_qp, x_ddp, xl, H, D,
        T_d0p, T_q0p, T_d0dp, T_q0dp, S1, S2,
        has_gov, pm_idx, has_exc, efd_idx, jac_pos)
end

# Phase 1.5: register with the device registry so build_layout! picks
# up Genrou without layout.jl having to know about it.
register_device!(:genrou;
    table_type = GenrouTable,
    builder    = _build_genrou_table_impl,
    class      = :generator)
