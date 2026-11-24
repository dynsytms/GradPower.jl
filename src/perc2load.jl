# PERC2 (Power Electronic Reconnecting and Ceasing, version 2) parameters.
#
# Names follow the `.dyr` `PERCxx2U` record (6 ICON flags + 57 CONs) so that
# reference data and `.dyr` records map one-to-one. The loading factor is
# `Ldfact` (also called `Lfm`). All per-unit quantities are on the model MVA
# base `Mbase = P0 / Ldfact`, except voltages (bus pu) and frequencies (Hz).
#
# Defaults reproduce the PERC1-standard behaviour: only the momentary
# low-voltage check is enabled.

"""
    PERC2Params(; kwargs...)

Settings of a [`PERC2Load`](@ref): the 6 cease-check flags and 57 constants
of the PERC2 `.dyr` record (`PERCBL2U`), in record order, plus three
GradPower options (`fnom`, `tt_floor_steps`, `use_qpratio`). Keywords not
given keep their defaults; the defaults enable only the momentary
low-voltage cease check (`VLMflag`).

Units: currents and powers in pu on the model base `Mbase = P0 / Ldfact`
(`P0`: initial active power of the PERC2 part); voltages in bus pu;
frequencies in Hz (`dbfl`, `dbfh` in pu of `fnom`); times in s.

- `fnom = 60.0`: nominal frequency (Hz), the same for every PERC2 load of a system.
- `tt_floor_steps = 2`: the output lag `Tt` is raised to `tt_floor_steps * dt`.
- `use_qpratio = true`: the dynamic reactive path starts at `QPratio * P0` and
  the rest of the power-flow `Q0` is a constant current that follows `FracOn`;
  `false` puts all of `Q0` on the dynamic path.

Parameters are checked by `add_perc2_load!` (for example `Imax = 0` and
negative time constants throw `ArgumentError`). The "Data-center loads
(PERC2)" guide lists every field with its unit and default.
"""
Base.@kwdef struct PERC2Params
    # flags (enabled when true)
    VLMflag::Bool = true
    VHMflag::Bool = false
    VLLflag::Bool = false
    VLHflag::Bool = false
    FLflag::Bool = false
    FHflag::Bool = false
    # base, filters, droop
    Ldfact::Float64 = 0.8
    QPratio::Float64 = 0.0
    Tt::Float64 = 0.02
    Tv::Float64 = 0.05
    Tf::Float64 = 0.02
    dbfl::Float64 = 0.0
    dbfh::Float64 = 0.0
    Kdroop::Float64 = 8.0
    # washouts and lead-lags
    Kvp::Float64 = 0.0
    Tvp::Float64 = 0.05
    Kvq::Float64 = 0.0
    Tvq::Float64 = 0.1
    Tap::Float64 = 1.0
    Tbp::Float64 = 1.0
    Taq::Float64 = 1.0
    Tbq::Float64 = 1.0
    # voltage exponents and current limits
    nP::Float64 = 0.0
    nQ::Float64 = 0.0
    Ipmax::Float64 = 1.0
    Ipmin::Float64 = 0.0
    Iqmax::Float64 = 0.7
    Iqmin::Float64 = -0.7
    Imax::Float64 = 1.5
    # momentary cease / reconnect
    fcease::Float64 = 0.9
    VLcease::Float64 = 0.5
    VHcease::Float64 = 1.2
    Tcease::Float64 = 0.05
    Tdelay::Float64 = 0.0
    frecon::Float64 = 0.5
    VLrecon::Float64 = 0.6
    VHrecon::Float64 = 1.1
    Tramp::Float64 = 1.25
    Trecon::Float64 = 0.05
    # long low-voltage cease
    TLrampL::Float64 = 1.25
    TLreconL::Float64 = 0.05
    VLC1::Float64 = 0.9
    TVLC1::Float64 = 2.0
    VLC2::Float64 = 0.8
    TVLC2::Float64 = 0.5
    VLC3::Float64 = 0.0
    TVLC3::Float64 = 0.0
    VLC4::Float64 = 0.5
    TVLC4::Float64 = 0.15
    # long high-voltage cease
    TLrampH::Float64 = 1.25
    TLreconH::Float64 = 0.05
    VHC1::Float64 = 1.1
    TVHC1::Float64 = 1.0
    VHC2::Float64 = 1.15
    TVHC2::Float64 = 0.5
    VHC3::Float64 = 99.0
    TVHC3::Float64 = 0.0
    VHC4::Float64 = 1.2
    TVHC4::Float64 = 0.16
    # frequency cease (never reconnects)
    FLC1::Float64 = 58.0
    TFLC1::Float64 = 100.0
    FHC1::Float64 = 62.0
    TFHC1::Float64 = 100.0
    # GradPower options (not `.dyr` CONs)
    fnom::Float64 = 60.0
    tt_floor_steps::Int = 2       # Tt is raised to tt_floor_steps*dt
    use_qpratio::Bool = true      # true: Mvarinit = QPratio*P0, rest of Q0 as constant Iqextra (block diagram);
                                  # false: Mvarinit = Q0, QPratio ignored (stand-alone reading)
end

const PERC2_FLAG_NAMES = (:VLMflag, :VHMflag, :VLLflag, :VLHflag, :FLflag, :FHflag)

const _PERC2_TIME_NAMES = (:Tt, :Tv, :Tf, :Tvp, :Tvq, :Tap, :Tbp, :Taq, :Tbq,
    :Tcease, :Tdelay, :Tramp, :Trecon, :TLrampL, :TLreconL,
    :TVLC1, :TVLC2, :TVLC3, :TVLC4, :TLrampH, :TLreconH,
    :TVHC1, :TVHC2, :TVHC3, :TVHC4, :TFLC1, :TFHC1)

"""
    validate(p::PERC2Params; dt=nothing) -> p

Throw `ArgumentError` for invalid parameter sets; return `p` otherwise.

Rejected:
- non-finite values, negative time constants, `tt_floor_steps < 0`, `fnom <= 0`;
- `Imax == 0` (it would zero every current limit, or could be read as
  "unused" — ambiguous, so it is refused; a negative `Imax` is used as `|Imax|`);
- `Ipmax < Ipmin` or `Iqmax < Iqmin`;
- a lead-lag with `Ta > 0` and `Tb == 0` (it would be bypassed silently);
- a washout with `K ≠ 0` and `T ≤ 0` (a pure derivative).

Accepted silently: `VLrecon < VLcease` (raised to `VLcease`),
`VHrecon > VHcease` (lowered to `VHcease`, whatever `VHMflag`), `fcease`
outside [0, 1] (clamped), negative `frecon` (treated as 0), `Ldfact <= 0`
(0.8 is used). Any finite `QPratio` is accepted: it sets
`Mvarinit = QPratio·P0` and the remainder of the power-flow Q becomes the
constant current `Iqextra` (see `perc2_init`).
"""
function validate(p::PERC2Params; dt = nothing)
    for n in fieldnames(PERC2Params)
        v = getfield(p, n)
        v isa AbstractFloat && !isfinite(v) &&
            throw(ArgumentError("PERC2: $n must be finite (got $v)"))
    end
    for n in _PERC2_TIME_NAMES
        getfield(p, n) < 0 && throw(ArgumentError("PERC2: time constant $n must be ≥ 0 (got $(getfield(p, n)))"))
    end
    p.tt_floor_steps >= 0 || throw(ArgumentError("PERC2: tt_floor_steps must be ≥ 0"))
    p.fnom > 0 || throw(ArgumentError("PERC2: fnom must be positive"))
    dt === nothing || dt > 0 || throw(ArgumentError("PERC2: dt must be positive (got $dt)"))
    p.Imax == 0 && throw(ArgumentError("PERC2: Imax = 0 zeroes every current limit; use a positive Imax"))
    p.Ipmax < p.Ipmin && throw(ArgumentError("PERC2: Ipmax < Ipmin"))
    p.Iqmax < p.Iqmin && throw(ArgumentError("PERC2: Iqmax < Iqmin"))
    p.Kvp != 0 && p.Tvp <= 0 && throw(ArgumentError("PERC2: Kvp ≠ 0 needs Tvp > 0 (sK/(1+sT) with T = 0 is a pure derivative)"))
    p.Kvq != 0 && p.Tvq <= 0 && throw(ArgumentError("PERC2: Kvq ≠ 0 needs Tvq > 0 (sK/(1+sT) with T = 0 is a pure derivative)"))
    p.Tap > 0 && p.Tbp == 0 && throw(ArgumentError("PERC2: Tap > 0 with Tbp = 0 (lead-lag would be bypassed)"))
    p.Taq > 0 && p.Tbq == 0 && throw(ArgumentError("PERC2: Taq > 0 with Tbq = 0 (lead-lag would be bypassed)"))
    return p
end

# ------------------------------------------------------------------------
# PERC2Load device (load class, no states)
# ------------------------------------------------------------------------

"""
    PERC2Load(bus, id, P0, Q0; params=PERC2Params())

PERC2 data-center load at external bus number `bus`, load id
`id`, consuming `P0 + j Q0` (system-base pu, `Q0 > 0` inductive) at the
power-flow voltage (`V0`, `va0`, set by `initialize_dynamics!`). It has no
differential or algebraic states: inside each Newton solve it draws the
system-base current `(Ip - j Iq) V/|V|` with `Ip`, `Iq` held in the parameter
vector; after every accepted step the PERC2 logic (`perc2_step!`) updates
them. Normally created by [`add_perc2_load!`](@ref), which splits an existing
static load so the power flow stays consistent.

Parameter vector (`get_param_names`), all read-only for `set_parameter!`:

| slot | name    | meaning                                                      |
|------|---------|--------------------------------------------------------------|
| 1    | `Ip`    | held active current drawn (system base), updated each step   |
| 2    | `Iq`    | held reactive current drawn (system base, > 0 inductive)     |
| 3    | `P0`    | initial active power (system-base pu)                        |
| 4    | `Q0`    | initial reactive power (system-base pu)                      |
| 5    | `V0`    | initial bus voltage magnitude (pu)                           |
| 6, 7 | `dir_r`, `dir_i` | `V/|V|` at the last accepted step, used when `|V| < 1e-6` |
"""
mutable struct PERC2Load <: AbstractLoadType
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64              # external bus number
    id::String
    params::PERC2Params
    P0::Float64
    Q0::Float64
    V0::Float64
    va0::Float64
end

const PERC2_PAR_SIZE = 7

function PERC2Load(bus::Integer, id::AbstractString, P0::Real, Q0::Real;
                   params::PERC2Params=PERC2Params(), V0::Real=1.0, va0::Real=0.0)
    return PERC2Load(0, 0, 0, PERC2_PAR_SIZE, Int64(bus), String(id), params,
                     Float64(P0), Float64(Q0), Float64(V0), Float64(va0))
end

function fill_pvec!(pvec::AbstractArray, dtype::PERC2Load)
    Ip, Iq = perc2_output(perc2_init(dtype.params, dtype.P0, dtype.Q0, dtype.V0))
    pvec[1] = Ip
    pvec[2] = Iq
    pvec[3] = dtype.P0
    pvec[4] = dtype.Q0
    pvec[5] = dtype.V0
    pvec[6] = cos(dtype.va0)
    pvec[7] = sin(dtype.va0)
end

get_device_name(::PERC2Load) = "PERC2Load"
get_bus(dtype::PERC2Load) = dtype.bus
get_param_names(::PERC2Load) = ["Ip", "Iq", "P0", "Q0", "V0", "dir_r", "dir_i"]
get_diff_names(::PERC2Load) = []
get_alg_names(::PERC2Load) = []

# Network entries only (self-bus 2x2 block, always in the pattern).
preallocate_jacobian!(coord_list::Vector{Vector{Int}}, diff_ptr::Int, alg_ptr::Int,
                      ctrl_ptr::Int, volt_ptr::Int, dtype::PERC2Load) = nothing

"""
    add_device!(sys::PowerSystem, dtype::PERC2Load) -> device index

Attach a PERC2 load to a system whose dynamics are already built
(`set_dynamics!`/`from_psse`). The load is NOT in the power flow: unless it
replaces part of an existing load (see [`add_perc2_load!`](@ref)) the initial
state is not an equilibrium. Re-create the `DynamicProblem` and run
`initialize_dynamics!` afterwards.
"""
function add_device!(ps::PowerSystem, dtype::PERC2Load)
    dyn = ps.dynamic
    (dyn === nothing || dyn.map === nothing || dyn.layout === nothing) &&
        throw(ArgumentError("add_device!: build the dynamic model (from_psse/set_dynamics!) before adding a PERC2Load"))
    haskey(ps.busmap, dtype.bus) || throw(ArgumentError("add_device!: bus $(dtype.bus) not in the case"))
    validate(dtype.params)
    add_device!(dyn, dtype)
    i = dyn.num_devices
    dm = dyn.map
    if i > length(dm.bus)
        for v in (dm.bus, dm.gen, dm.load, dm.diff_ptr, dm.alg_ptr, dm.ctrl_ptr, dm.par_ptr,
                  dm.diff_size, dm.alg_size, dm.ctrl_size, dm.par_size)
            push!(v, 0)
        end
    end
    dev = dyn.devices[i]
    dm.bus[i] = ps.busmap[dtype.bus]
    dm.gen[i] = 0
    dm.load[i] = 0
    dm.diff_ptr[i] = dev.diff_ptr
    dm.alg_ptr[i] = dev.alg_ptr
    dm.ctrl_ptr[i] = dev.ctrl_ptr
    dm.par_ptr[i] = dev.par_ptr
    dm.diff_size[i] = 0
    dm.alg_size[i] = 0
    dm.ctrl_size[i] = 0
    dm.par_size[i] = dtype.par_size
    # No states: z, clusters and the other tables are unchanged; only the
    # PERC2 table is rebuilt.
    L = dyn.layout::SimulationLayout
    dyn.layout = SimulationLayout(merge(L.tables, (perc2load = _build_perc2load_table_impl(dyn),)),
                                  L.net_jac_pos, L.diff_row_pos, L.diff_diag_pos)
    return i
end

"""
    add_perc2_load!(sys, bus_ext::Int, load_id::AbstractString;
                    fraction=1.0, params=PERC2Params()) -> device index

Replace `fraction` (0 < fraction ≤ 1) of the static load `load_id` at external bus
`bus_ext` with a [`PERC2Load`](@ref); the remaining `1 - fraction` stays in the
load's `ZIPLoad` (its constant-power and constant-impedance parts are scaled;
with `fraction = 1` the ZIPLoad stays in place with zero power, so device
indices do not shift). Returns the index of the new device in
`sys.dynamic.devices`.

The PERC2 initial consumption is `fraction * (pd, -qd)` of the static `Load`
(system-base pu; `Load.qd` carries the injection sign) and its initial voltage
the power-flow voltage of the bus (refreshed by `initialize_dynamics!`). The
power flow itself is unchanged, so the split system starts in equilibrium.

Order: `from_psse` → `build_network!` → `runpf!` → `add_perc2_load!` →
`DynamicProblem` → `initialize_dynamics!` → `integrate!`. A `DynamicProblem`
created before the call has the wrong parameter length; `integrate!` throws.
Each static load can be split once.

Notes:
- PERC2 has no entries in `z`. Use `integrate!(...; perc2_log = v)` or
  [`perc2_states`](@ref) to read its quantities.
- Every `integrate!` restarts the PERC2 logic from its initial operating
  point at the first time of the run (timers are absolute times).
- `set_load!`, `LoadStepEvent` and `add_load_trip!` on this load act on the
  ZIPLoad part only; `set_load!` overwrites the split — call it before
  `add_perc2_load!`. Trip the PERC2 part with
  `DisconnectDeviceEvent(find_device(sys, :perc2load, bus_ext, load_id), t)`.
- Supported: the CPU `integrate!` with `solver=:monolithic`, both
  integrators (`:backward_euler`, `:crank_nicolson`) and every
  `limit_method` (these act on TGOV1 limits; the PERC2 current limits are
  internal to the model). `solver=:schur`/`:schur_gmres`, GPU/KernelAbstractions
  integrators and `DynamicNLP` throw `ArgumentError`. Batched CPU
  integration (`integrate_batched!`) runs one PERC2 logic per scenario.
"""
function add_perc2_load!(ps::PowerSystem, bus_ext::Int, load_id::AbstractString;
                         fraction::Real=1.0, params::PERC2Params=PERC2Params())
    dyn = ps.dynamic
    (dyn === nothing || dyn.map === nothing) &&
        throw(ArgumentError("add_perc2_load!: no dynamic model; build the system with from_psse/set_dynamics! first"))
    0.0 < fraction <= 1.0 ||
        throw(ArgumentError("add_perc2_load!: fraction must be in (0, 1], got $fraction"))
    di = find_load_device(ps, bus_ext, load_id)
    li = dyn.map.load[di]
    li == 0 && throw(ArgumentError("add_perc2_load!: load \"$load_id\" at bus $bus_ext has no static Load"))
    key = _normalize_id(load_id)
    any(d -> d.dtype isa PERC2Load && d.dtype.bus == bus_ext && _normalize_id(d.dtype.id) == key,
        dyn.devices) &&
        throw(ArgumentError("add_perc2_load!: load \"$key\" at bus $bus_ext already has a PERC2 part"))
    load = ps.loads[li]
    zl = dyn.devices[di].dtype::ZIPLoad
    P0 = fraction * load.pd
    Q0 = -fraction * load.qd
    b = ps.busmap[bus_ext]
    dev = PERC2Load(bus_ext, key, P0, Q0; params=params, V0=ps.buses[b].v0m, va0=ps.buses[b].v0a)
    perc2_init(params, P0, Q0, dev.V0)       # throws on invalid parameters / operating point

    keep = 1.0 - fraction
    zl.pinj = keep * load.pd
    zl.qinj = keep * load.qd
    # yreal/yimag (and pvec) are derived from pinj/qinj by initialize_dynamics!.
    L = dyn.layout
    if L !== nothing
        k = count(d -> d.dtype isa ZIPLoad, view(dyn.devices, 1:di))
        L.zipload.pinj[k] = zl.pinj
        L.zipload.qinj[k] = zl.qinj
    end
    return add_device!(ps, dev)
end
