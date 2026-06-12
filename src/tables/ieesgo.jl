# Phase 1 of ROADMAP.md (agent A1.2): builder for IEESGOTable.
#
# This file is included from src/GradPower.jl AFTER both layout.jl (for the
# IEESGOTable type) and dynamics.jl (which transitively includes
# governors.jl for the IEESGO struct).
#
# The stub `build_ieesgo_table` defined in src/layout.jl delegates to
# `_build_ieesgo_table_impl` below.

"""
    _build_ieesgo_table_impl(psd) -> IEESGOTable

Scan `psd.devices` once and emit a struct-of-arrays table for every IEESGO
governor. All vectors have length `n` = number of IEESGO devices.

`w_idx[k]` is the global z-index of the generator's `w` state that governor
`k` reads. This is resolved from `psd.uvec_idx[ctrl_ptr]`, which
`set_dynamics!` populates BEFORE `build_layout!` is called. If a governor
has no matching generator (orphan in `.dyr`), `psd.uvec_idx[ctrl_ptr] == 0`
and `w_idx[k] = 0`; Phase 2 kernels are expected to skip these.

`jac_pos` is allocated as an `n × 0` Int32 matrix; Phase 2 widens its
second dimension via `preallocate_jacobian`.
"""
function _build_ieesgo_table_impl(psd)
    # 1. Count IEESGO governors.
    n = 0
    for device in psd.devices
        if device.dtype isa IEESGO
            n += 1
        end
    end

    # 2. Allocate global pointer vectors.
    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    alg_ptr  = Vector{Int32}(undef, n)
    ctrl_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    # 3. Allocate parameter vectors (11 IEESGO parameters).
    T1   = Vector{Float64}(undef, n)
    T2   = Vector{Float64}(undef, n)
    T3   = Vector{Float64}(undef, n)
    T4   = Vector{Float64}(undef, n)
    T5   = Vector{Float64}(undef, n)
    T6   = Vector{Float64}(undef, n)
    K1   = Vector{Float64}(undef, n)
    K2   = Vector{Float64}(undef, n)
    K3   = Vector{Float64}(undef, n)
    pmax = Vector{Float64}(undef, n)
    pmin = Vector{Float64}(undef, n)
    # pref is filled with 0 at build time; refresh_ieesgo_table! mirrors
    # the post-init value from initialize_dynamics!.
    pref = zeros(Float64, n)

    # 4. Control-coupling vector: global z-index of generator's w state.
    w_idx = Vector{Int32}(undef, n)

    # 5. Phase 2.1: jac_pos has 16 entries per IEESGO device. See
    #    src/kernels/ieesgo.jl IEESGO_JAC_NENTRIES.
    jac_pos = zeros(Int32, n, IEESGO_JAC_NENTRIES)

    # 6. Single pass over devices, fill row k for each IEESGO match.
    uvec = psd.uvec_idx
    k = 0
    for device in psd.devices
        device.dtype isa IEESGO || continue
        k += 1
        gov = device.dtype

        bus[k]      = Int32(gov.bus)
        diff_ptr[k] = Int32(device.diff_ptr)
        alg_ptr[k]  = Int32(device.alg_ptr)
        ctrl_ptr[k] = Int32(device.ctrl_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        T1[k]   = gov.T1
        T2[k]   = gov.T2
        T3[k]   = gov.T3
        T4[k]   = gov.T4
        T5[k]   = gov.T5
        T6[k]   = gov.T6
        K1[k]   = gov.K1
        K2[k]   = gov.K2
        K3[k]   = gov.K3
        pmax[k] = gov.pmax
        pmin[k] = gov.pmin

        # Control coupling: ctrl_ptr points at u[1] (w). uvec_idx[ctrl_ptr] == 0
        # means the governor was not paired to a generator (orphan in .dyr).
        w_routed = uvec[device.ctrl_ptr]
        w_idx[k] = Int32(w_routed)
    end

    return IEESGOTable(n, bus, diff_ptr, alg_ptr, ctrl_ptr, par_ptr,
        T1, T2, T3, T4, T5, T6, K1, K2, K3, pmax, pmin, pref,
        w_idx, jac_pos)
end

"""
    refresh_ieesgo_table!(psd)

Re-snapshot the `pref` column from device structs after
`initialize_dynamics!` has converged. Mirror of `refresh_zipload_table!`
in src/tables/zipload.jl. Called from `initialize_dynamics!` after the
controller-init loop.
"""
function refresh_ieesgo_table!(psd)
    table = psd.layout.ieesgo
    table.n == 0 && return nothing
    k = 0
    for device in psd.devices
        device.dtype isa IEESGO || continue
        k += 1
        table.pref[k] = device.dtype.pref
    end
    @assert k == table.n
    return nothing
end

# Phase 1.5: register with the device registry.
register_device!(:ieesgo;
    table_type = IEESGOTable,
    builder    = _build_ieesgo_table_impl,
    class      = :governor)
