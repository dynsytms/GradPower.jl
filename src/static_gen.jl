# StaticGenerator — aggregated representation of a bus's active static
# generators that lack a dynamic machine model.
#   - PV (bus type 2):    Q is unknown (alg state); P is the parameter.
#                         Voltage-regulation residual: vr^2 + vi^2 = vset^2.
#   - SLACK (bus type 3): P and Q are both unknown (alg states).
#                         Voltage angle/magnitude pinned to (vset, aset).
#   - PQ (bus type 1):    P and Q are both parameters. No alg states; pure
#                         current injection into the voltage rows.
#
# Current injection (all bus types) is:
#   f[vr] += (p*vr + q*vi) / vm2
#   f[vi] += (p*vi - q*vr) / vm2
# with vm2 saturated at 0.2 for numerical stability — same threshold as
# ZIPLoad.
#
# A StaticGenerator is bound to ONE bus and aggregates one or more static
# generators at that bus. `gen_idxs` indexes into ps.gens.

mutable struct StaticGenerator <: AbstractDeviceType
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    # topology
    bus::Int64                # external PSS/E bus number (becomes internal via busmap during build_layout!)
    bus_type::Int64           # 1=PQ, 2=PV, 3=SLACK
    gen_idxs::Vector{Int64}   # indices into ps.gens that this device aggregates
    # parameters / setpoints
    vset::Float64
    aset::Float64
    # initialization-derived (filled by initialize_static_gens!)
    p0::Float64
    q0::Float64
end

function StaticGenerator(bus::Int64, bus_type::Int64, gen_idxs::Vector{Int64},
                          vset::Float64, aset::Float64)
    alg_size = bus_type == 2 ? 1 : (bus_type == 3 ? 2 : 0)
    par_size = 4   # [p, q, vset, aset]
    return StaticGenerator(0, alg_size, 0, par_size, bus, bus_type, gen_idxs,
                           vset, aset, 0.0, 0.0)
end

function fill_pvec!(pvec::AbstractArray, dtype::StaticGenerator)
    pvec[1] = dtype.p0
    pvec[2] = dtype.q0
    pvec[3] = dtype.vset
    pvec[4] = dtype.aset
end

get_device_name(::StaticGenerator) = "StaticGenerator"
get_bus(dtype::StaticGenerator) = dtype.bus
get_param_names(::StaticGenerator) = ["p", "q", "vset", "aset"]
get_diff_names(::StaticGenerator) = String[]
get_alg_names(dtype::StaticGenerator) =
    dtype.bus_type == 2 ? ["q"] : (dtype.bus_type == 3 ? ["p", "q"] : String[])
