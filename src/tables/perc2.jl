# SoA table for PERC2Load (data-center load, partitioned formulation).
#
# Like ZIPLoad, a PERC2Load has no diff/alg/ctrl states and only writes the
# self-bus 2x2 block of the voltage rows. The held currents Ip, Iq live in the
# parameter vector (slots 1, 2), so every Newton path reads them through `p`.
#
# The PERC2 logic state (filters, timers, cease/reconnect machine, Tt lag) is
# not part of z or p: it is advanced once per ACCEPTED step by
# `perc2_accept_step!` (src/kernels/perc2.jl). It is kept per scenario in
# `states[m]` / `freq[m]` (scenario 1 is used by `integrate!`), so batched
# integrators can keep one store per scenario.

const PERC2_JAC_NENTRIES = 4

struct PERC2LoadTable
    n::Int
    bus::Vector{Int32}                 # internal bus index
    par_ptr::Vector{Int32}
    device::Vector{Int32}              # index in psd.devices
    params::Vector{PERC2Params}
    jac_pos::Matrix{Int32}
    online::Vector{Bool}
    # per-scenario logic store (outside z and p)
    states::Vector{Vector{PERC2State}}           # states[m][k]
    freq::Vector{BusFrequencyEstimator}          # freq[m], buses = bus
    last_vm::Vector{Vector{Float64}}             # |V| at the last accepted step
end

"""
    _build_perc2load_table_impl(psd) -> PERC2LoadTable

Scan `psd.devices` for `PERC2Load`s. The internal bus index comes from
`psd.map.bus` (the device stores the external bus number). The per-scenario logic
store starts with one (empty) scenario; it is filled by `perc2_init_scenario!`.
"""
function _build_perc2load_table_impl(psd)
    idx = [i for (i, d) in enumerate(psd.devices) if d.dtype isa PERC2Load]
    n = length(idx)
    bus = Vector{Int32}(undef, n)
    par_ptr = Vector{Int32}(undef, n)
    params = Vector{PERC2Params}(undef, n)
    for (k, i) in enumerate(idx)
        dev = psd.devices[i]
        bus[k] = Int32(psd.map === nothing ? dev.dtype.bus : psd.map.bus[i])
        par_ptr[k] = Int32(dev.par_ptr)
        params[k] = dev.dtype.params
    end
    return PERC2LoadTable(n, bus, par_ptr, Int32.(idx), params,
                          zeros(Int32, n, PERC2_JAC_NENTRIES), fill(true, n),
                          [PERC2State[]], [BusFrequencyEstimator(Int.(bus))], [zeros(n)])
end

"""
    refresh_perc2_table!(psd)

Re-snapshot the PERC2 parameters (`params`) from the device structs. Called
by `initialize_dynamics!`.
"""
function refresh_perc2_table!(psd)
    table = psd.layout.perc2load
    table.n == 0 && return nothing
    for k in 1:table.n
        table.params[k] = psd.devices[table.device[k]].dtype.params
    end
    return nothing
end

register_device!(:perc2load;
    table_type = PERC2LoadTable,
    builder    = _build_perc2load_table_impl,
    class      = :load)
