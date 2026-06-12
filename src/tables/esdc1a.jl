# Phase 1 of ROADMAP.md (agent A1.3): builder for ESDC1ATable.
#
# This file is included from src/GradPower.jl AFTER layout.jl (for the
# ESDC1ATable type). Note: as of this writing, src/exciters.jl (which
# defines the ESDC1A struct) is UNTRACKED in git and NOT included anywhere
# in the live module. No live case (.dyr) instantiates ESDC1A, so this
# builder always returns an empty table on every currently-validated case.
#
# To remain correct whether or not the ESDC1A type is loaded, we duck-type
# the match by the type's `nameof` rather than `isa ESDC1A`.
#
# The stub `build_esdc1a_table` defined in src/layout.jl delegates to
# `_build_esdc1a_table_impl` below.

# Duck-typed predicate: matches devices whose dtype's typename is `:ESDC1A`.
# Works whether or not the ESDC1A type is in scope at compile time.
is_esdc1a(d) = nameof(typeof(d.dtype)) === :ESDC1A

"""
    _build_esdc1a_table_impl(psd) -> ESDC1ATable

Scan `psd.devices` once and emit a struct-of-arrays table for every ESDC1A
exciter. All vectors have length `n` = number of ESDC1A devices.

ESDC1A has 3 diff states (vr1, vr2, e_fd), 0 alg states, 0 ctrl, and 10
parameters (Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be, vref). Parameter order
matches the field order of ESDC1ATable in src/layout.jl.

`jac_pos` is allocated as an `n × 0` Int32 matrix; Phase 2 widens its
second dimension via `preallocate_jacobian`.
"""
function _build_esdc1a_table_impl(psd)
    # 1. Count ESDC1A exciters (duck-typed).
    n = 0
    for device in psd.devices
        if is_esdc1a(device)
            n += 1
        end
    end

    # 2. Allocate global pointer vectors. ESDC1A has no alg / no ctrl,
    #    so ESDC1ATable carries no alg_ptr / ctrl_ptr.
    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    # 3. Allocate parameter vectors (10 ESDC1A parameters).
    Ka   = Vector{Float64}(undef, n)
    Ta   = Vector{Float64}(undef, n)
    Kf   = Vector{Float64}(undef, n)
    Tf   = Vector{Float64}(undef, n)
    Ke   = Vector{Float64}(undef, n)
    Te   = Vector{Float64}(undef, n)
    Tr   = Vector{Float64}(undef, n)
    Ae   = Vector{Float64}(undef, n)
    Be   = Vector{Float64}(undef, n)
    vref = Vector{Float64}(undef, n)

    # 4. Phase 2 will fill jac_pos's second dim; Phase 1 leaves it empty.
    jac_pos = Matrix{Int32}(undef, n, 0)

    # 5. Single pass over devices, fill row k for each ESDC1A match.
    #    Use getfield since d.dtype is of an unknown-at-compile-time type
    #    from this file's perspective (ESDC1A is not in scope).
    k = 0
    for device in psd.devices
        is_esdc1a(device) || continue
        k += 1
        exc = device.dtype

        bus[k]      = Int32(getfield(exc, :bus))
        diff_ptr[k] = Int32(device.diff_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        Ka[k]   = getfield(exc, :Ka)
        Ta[k]   = getfield(exc, :Ta)
        Kf[k]   = getfield(exc, :Kf)
        Tf[k]   = getfield(exc, :Tf)
        Ke[k]   = getfield(exc, :Ke)
        Te[k]   = getfield(exc, :Te)
        Tr[k]   = getfield(exc, :Tr)
        Ae[k]   = getfield(exc, :Ae)
        Be[k]   = getfield(exc, :Be)
        vref[k] = getfield(exc, :vref)
    end

    return ESDC1ATable(n, bus, diff_ptr, par_ptr,
        Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be, vref,
        jac_pos)
end

# Phase 1.5: register with the device registry.
register_device!(:esdc1a;
    table_type = ESDC1ATable,
    builder    = _build_esdc1a_table_impl,
    class      = :exciter)
