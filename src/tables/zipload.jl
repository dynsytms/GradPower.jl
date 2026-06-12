# Phase 1 of ROADMAP.md (agent A1.4): builder for ZIPLoadTable.
#
# This file is included from src/GradPower.jl AFTER both layout.jl (for the
# ZIPLoadTable type) and dynamics.jl (which transitively includes
# loads.jl for the ZIPLoad struct).
#
# The stub `build_zipload_table` defined in src/layout.jl delegates to
# `_build_zipload_table_impl` below.

"""
    _build_zipload_table_impl(psd) -> ZIPLoadTable

Scan `psd.devices` once and emit a struct-of-arrays table for every ZIPLoad.
All vectors have length `n` = number of ZIPLoad devices.

ZIPLoad has 0 diff / 0 alg / 0 ctrl / 9 params; it is purely algebraic via
`cinject!` into the voltage rows. Hence only `bus` and `par_ptr` pointers
are stored — no diff/alg/ctrl pointers.

NOTE on snapshot timing: `build_layout!` runs at the END of `set_dynamics!`,
which is BEFORE `initialize_dynamics!` mutates `dtype.v0mag`, `dtype.yreal`,
`dtype.yimag` from the power-flow solution. At build time, `yreal` and
`yimag` are typically still 0.0 (from the ZIPLoad constructor in
`set_dynamics!`). Phase 1 snapshots whatever is in the struct now; Phase 2
will decide whether to re-snapshot post-init or to read live values from
`dp.pvec` in the hot loop.

`jac_pos` is allocated as an `n × 0` Int32 matrix; Phase 2 widens its
second dimension via `preallocate_jacobian`.
"""
function _build_zipload_table_impl(psd)
    # 1. Count ZIPLoads.
    n = 0
    for device in psd.devices
        if device.dtype isa ZIPLoad
            n += 1
        end
    end

    # 2. Allocate global pointer vectors. ZIPLoad has no diff/alg/ctrl
    #    state, so only `bus` and `par_ptr` are needed.
    bus     = Vector{Int32}(undef, n)
    par_ptr = Vector{Int32}(undef, n)

    # 3. Allocate parameter vectors (9 ZIPLoad parameters).
    pinj   = Vector{Float64}(undef, n)
    qinj   = Vector{Float64}(undef, n)
    α      = Vector{Float64}(undef, n)
    β      = Vector{Float64}(undef, n)
    γ      = Vector{Float64}(undef, n)
    weight = Vector{Float64}(undef, n)
    v0mag  = Vector{Float64}(undef, n)
    yreal  = Vector{Float64}(undef, n)
    yimag  = Vector{Float64}(undef, n)

    # 4. Phase 2 will fill jac_pos's second dim; Phase 1 leaves it empty.
    jac_pos = Matrix{Int32}(undef, n, 0)

    # 5. Single pass over devices, fill row k for each ZIPLoad match.
    k = 0
    for device in psd.devices
        device.dtype isa ZIPLoad || continue
        k += 1
        load = device.dtype

        bus[k]     = Int32(load.bus)
        par_ptr[k] = Int32(device.par_ptr)

        pinj[k]   = load.pinj
        qinj[k]   = load.qinj
        α[k]      = load.α
        β[k]      = load.β
        γ[k]      = load.γ
        weight[k] = load.weight
        v0mag[k]  = load.v0mag
        yreal[k]  = load.yreal
        yimag[k]  = load.yimag
    end

    return ZIPLoadTable(n, bus, par_ptr,
        pinj, qinj, α, β, γ, weight, v0mag, yreal, yimag,
        jac_pos)
end

"""
    refresh_zipload_table!(psd)

Re-snapshot the ZIPLoad columns (`v0mag`, `yreal`, `yimag`) from the
current `device.dtype` fields. Called by `initialize_dynamics!` AFTER
the power-flow solution has been used to populate `yreal`/`yimag` from
`pinj/(v0mag^2)`, so the SoA table reflects the live values the hot
loop kernels will read.

This is Phase 2.0c (ROADMAP §3): `build_layout!` runs at the end of
`set_dynamics!`, BEFORE `initialize_dynamics!` populates these fields,
so the initial snapshot is all zeros. Hot-loop kernels in Phase 2.1 read
from the table (not from `dp.pvec`), so the table must be refreshed
before the first Newton iteration.

Other ZIPLoad columns (`pinj`, `qinj`, `α`, `β`, `γ`, `weight`) are not
mutated by `initialize_dynamics!` and don't need refreshing.
"""
function refresh_zipload_table!(psd)
    table = psd.layout.zipload
    table.n == 0 && return nothing
    k = 0
    for device in psd.devices
        device.dtype isa ZIPLoad || continue
        k += 1
        load = device.dtype
        table.v0mag[k] = load.v0mag
        table.yreal[k] = load.yreal
        table.yimag[k] = load.yimag
    end
    @assert k == table.n "refresh_zipload_table!: counted $k ZIPLoads, table.n == $(table.n)"
    return nothing
end

# Phase 1.5: register with the device registry.
register_device!(:zipload;
    table_type = ZIPLoadTable,
    builder    = _build_zipload_table_impl,
    class      = :load)
