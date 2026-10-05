# Generic SoA table + build-time plumbing for the `AbstractStdExciter`
# models (src/exciters_psse.jl): IEEET1, EXPIC1, EXAC2, EXAC1, ESAC1A, SCRX,
# ESAC6A.
#
# All of these exciters have the same table shape — diff states only, a
# terminal-voltage index, an optional PSS input index — so one parametric
# table type serves them all. `StdExcTable{M}` is CONCRETE for each model
# type M, so the SimulationLayout NamedTuple stays concrete and every kernel
# specializes (see the performance note in src/layout.jl).
#
# The kernels read parameters from the flat `p` vector through `par_ptr`, like
# every other device.

struct StdExcTable{M}
    n::Int
    bus::Vector{Int32}          # EXTERNAL bus number (as in the .dyr)
    diff_ptr::Vector{Int32}     # absolute z-index of the first diff state
    par_ptr::Vector{Int32}
    vr_idx::Vector{Int32}       # z-index of v_re at the terminal bus (fix_std_exc_vr_idx!)
    vs_idx::Vector{Int32}       # PSS v_s z-index; 0 = no PSS attached
    jac_pos::Matrix{Int32}
    online::Vector{Bool}
end

# Model registry for the generic plumbing. Each entry is the model type; the
# table name is `std_exc_name(M)`. Keep in sync with the register_device!
# calls at the bottom of this file.
std_exc_name(::Type{IEEET1}) = :ieeet1
std_exc_name(::Type{EXPIC1}) = :expic1
std_exc_name(::Type{EXAC2})  = :exac2
std_exc_name(::Type{EXAC1})  = :exac1
std_exc_name(::Type{ESAC1A}) = :esac1a
std_exc_name(::Type{SCRX})   = :scrx
std_exc_name(::Type{ESAC6A}) = :esac6a

const STD_EXC_TYPES = (IEEET1, EXPIC1, EXAC2, EXAC1, ESAC1A, SCRX, ESAC6A)

_std_exc_table(L, ::Type{M}) where {M} = getproperty(L, std_exc_name(M))

function _build_std_exc_table(::Type{M}, psd, njac::Int) where {M}
    n = 0
    for device in psd.devices
        device.dtype isa M && (n += 1)
    end
    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)
    k = 0
    for device in psd.devices
        device.dtype isa M || continue
        k += 1
        bus[k]      = Int32(device.dtype.bus)
        diff_ptr[k] = Int32(device.diff_ptr)
        par_ptr[k]  = Int32(device.par_ptr)
    end
    return StdExcTable{M}(n, bus, diff_ptr, par_ptr,
                          zeros(Int32, n), zeros(Int32, n),
                          zeros(Int32, n, njac), fill(true, n))
end

# External -> internal bus numbering for the terminal-voltage index. Called
# from set_dynamics! after build_layout! (like fix_esst4b_vr_idx!).
function fix_std_exc_vr_idx!(psd, ps)
    net_ptr = psd.diff_dim + psd.alg_dim
    for M in STD_EXC_TYPES
        table = _std_exc_table(psd.layout, M)
        table.n == 0 && continue
        for k in 1:table.n
            internal_bus = ps.busmap[Int(table.bus[k])]
            table.vr_idx[k] = Int32(net_ptr + 2*(internal_bus - 1) + 1)
        end
    end
    return nothing
end

# IEEEST v_s -> exciter wiring for the std exciters. Same matching rule as
# `fix_ieeest_wiring!` (src/tables/ieeest.jl): the PSS output is its first
# alg state, matched to the exciter by (bus, normalized id). Called from
# set_dynamics! right after fix_ieeest_wiring!. Always on, like ESST4B's
# (ESST4B_WIRE_PSS, src/exciters.jl).
function fix_std_exc_pss_wiring!(psd, ps)
    L = psd.layout
    L.ieeest.n == 0 && return nothing
    diff_dim = psd.diff_dim
    for device in psd.devices
        device.dtype isa IEEEST || continue
        pss_bus = device.dtype.bus
        pss_id  = _normalize_id(device.dtype.id)
        vs_z = diff_dim + device.alg_ptr
        for M in STD_EXC_TYPES
            table = _std_exc_table(L, M)
            table.n == 0 && continue
            k = 0
            for d in psd.devices
                d.dtype isa M || continue
                k += 1
                if d.dtype.bus == pss_bus && _normalize_id(d.dtype.id) == pss_id
                    table.vs_idx[k] = Int32(vs_z)
                end
            end
        end
    end
    return nothing
end

# Cluster reorder: diff_ptr is absolute; vr_idx is a network index (> n_da)
# and stays; vs_idx points at an IEEEST alg state and is remapped.
function _remap_std_exciters!(layout, old_to_new::Vector{Int}, diff_dim::Int, n_da::Int)
    for M in STD_EXC_TYPES
        table = _std_exc_table(layout, M)
        for k in 1:table.n
            table.diff_ptr[k] = Int32(old_to_new[Int(table.diff_ptr[k])])
            old_vs = Int(table.vs_idx[k])
            if old_vs > 0 && old_vs <= n_da
                table.vs_idx[k] = Int32(old_to_new[old_vs])
            end
        end
    end
    return nothing
end

_set_table_online!(L::SimulationLayout, d::AbstractStdExciter, k::Int, v::Bool) =
    (_std_exc_table(L, typeof(d)).online[k] = v; nothing)

_device_symbol(d::AbstractStdExciter) = std_exc_name(typeof(d))

register_device!(:ieeet1;
    table_type = StdExcTable{IEEET1},
    builder    = psd -> _build_std_exc_table(IEEET1, psd, IEEET1_JAC_NENTRIES),
    class      = :exciter)

register_device!(:expic1;
    table_type = StdExcTable{EXPIC1},
    builder    = psd -> _build_std_exc_table(EXPIC1, psd, EXPIC1_JAC_NENTRIES),
    class      = :exciter)

register_device!(:exac2;
    table_type = StdExcTable{EXAC2},
    builder    = psd -> _build_std_exc_table(EXAC2, psd, EXAC2_JAC_NENTRIES),
    class      = :exciter)

# EXAC1 / ESAC1A reuse the EXAC2 kernels and slot count.
register_device!(:exac1;
    table_type = StdExcTable{EXAC1},
    builder    = psd -> _build_std_exc_table(EXAC1, psd, EXAC2_JAC_NENTRIES),
    class      = :exciter)

register_device!(:esac1a;
    table_type = StdExcTable{ESAC1A},
    builder    = psd -> _build_std_exc_table(ESAC1A, psd, EXAC2_JAC_NENTRIES),
    class      = :exciter)

register_device!(:scrx;
    table_type = StdExcTable{SCRX},
    builder    = psd -> _build_std_exc_table(SCRX, psd, SCRX_JAC_NENTRIES),
    class      = :exciter)

register_device!(:esac6a;
    table_type = StdExcTable{ESAC6A},
    builder    = psd -> _build_std_exc_table(ESAC6A, psd, ESAC6A_JAC_NENTRIES),
    class      = :exciter)
