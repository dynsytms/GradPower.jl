using Match

include("parse_psse_raw.jl")
# ======================
# MATPOWER (*.m) parser
# ======================

function parse_matpower_line(line, field_names)
    split_line = split(line)
    parsed_line = map(x -> tryparse(Float64, x), split_line)
    return Dict(zip(field_names, parsed_line))
end

function parse_matpower_data(block_data, field_names)
    lines = split(block_data, '\n')
    lines = filter(line -> !isempty(strip(line)), lines)  # remove empty lines
    data = map(line -> parse_matpower_line(line, field_names), lines)
    return data
end

function read_matpower_case(file_name)
    file_content = read(file_name, String)
    mpc = Dict()

    # Parse single value entries
    for block in ["version", "baseMVA"]
        match_block = match(Regex(block * "\\s=\\s(.*?);", "s"), file_content)
        if match_block !== nothing
            value = strip(match_block.captures[1])
            if block == "version"
                mpc[block] = value
            else
                mpc[block] = parse(Float64, value)
            end
        end
    end

    # Define fields for each data block
    bus_fields = [
        "bus_i", "type", "Pd", "Qd", "Gs", "Bs", "area", "Vm", "Va", 
        "baseKV", "zone", "Vmax", "Vmin"
    ]
    gen_fields = [
        "bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", 
        "Pmax", "Pmin", "Pc1", "Pc2", "Qc1min", "Qc1max", "Qc2min", 
        "Qc2max", "ramp_agc", "ramp_10", "ramp_30", "ramp_q", "apf"
    ]
    branch_fields = [
        "fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC", 
        "ratio", "angle", "status", "angmin", "angmax"
    ]

    # Parse data blocks
    for (block, fields) in [
        ("bus", bus_fields), 
        ("gen", gen_fields), 
        ("branch", branch_fields)
    ]
        block_regex = Regex(block * "\\s=\\s\\[(.*?)\\];", "s")
        match_block = match(block_regex, file_content)
        if match_block !== nothing
            block_data = match_block.captures[1]
            mpc[block] = parse_matpower_data(block_data, fields)
        end
    end

    return mpc
end


# ============================
# PSSE Parser and Constructor
# ============================

"""
    from_psse(raw_data_file::String, dyr_file::String)

Reads a PSSE raw file and a PSSE dyr file and constructs a PowerSystem

"""
function from_psse(raw_file::String, dyr_file::Union{String, Nothing};
                    add_static_gen_stubs::Bool=true,
                    surrogates::Bool=false)
    raw = read_psse_raw(raw_file)
    sys = raw_to_grad(raw)
    if dyr_file !== nothing
        # Drop GENROU/GENSAL dynamic rows whose (bus, id) doesn't
        # reference an active static generator. The raw_to_grad pass has
        # already filtered status==0 gens, so the static `sys.gens` is
        # exactly the active set.
        active = Set{Tuple{Int64,String}}()
        for gen in sys.gens
            push!(active, (sys.buses[gen.bus].i, _normalize_id(gen.id)))
        end
        psd = PowerSystemDynamics(dyr_file; active_gen_keys=active, surrogates=surrogates)
        set_dynamics!(sys, psd; add_static_gen_stubs=add_static_gen_stubs)
    end
    return sys
end

# ======================
#  PSSE Dynamics (*.dyr)
# ======================

const DEVICE_TYPE_MAP = Dict(
    "GENROU" => Genrou,
    "GENSAL" => Gensal,
    "IEESGO" => IEESGO,
    "TGOV1"  => TGOV1,
    "SEXS"   => SEXS,
    "ESDC1A" => ESDC1A,
    "IEEEST" => IEEEST,
    # add more device types here
)

# ---------------------------------------------------------------------------
# Compatibility surrogates
# ---------------------------------------------------------------------------
#
# A surrogate maps a DYR model GradPower has no native kernel for onto a model
# it does implement. This is what lets a case like ACTIVSg2000 run end to end
# without first implementing a dozen controller models -- but a surrogate is a
# STAND-IN, not an implementation: it does not reproduce the source model's
# equations, and it must never be counted as native coverage or used for
# scientific validation (plan_enhance.md sections 2.2 and 7.1).
#
# Surrogates are OPT-IN. `from_psse(...; surrogates=true)` enables them, and
# every redirected record is reported distinctly from native ones.
#
# Constants for GGOV1, EXPIC1, SCRX and ESAC6A are taken from uqgrid's own
# compatibility redirects (uqgrid/uqgrid/io/parse.py) so the two simulators
# agree. `R` is passed on machine base; `set_ratio!` converts it to system
# base, matching uqgrid's `R * basemva / mbase`.
#
# The remaining entries are GradPower-local: uqgrid implements those natively,
# so there is no upstream mapping to copy. They reuse the same surrogate shape.

_f(fields, i) = parse(Float64, fields[i])

# uqgrid's SEXS surrogate: a plain fast AVR with effectively no output limit.
_sexs_surrogate(bus, id) = SEXS(bus, id, 0.4, 5.0, 20.0, 1.0, -99.0, 99.0)

# uqgrid's TGOV1 surrogate shape; only the droop R comes from the record.
_tgov1_surrogate(bus, id, R) = TGOV1(bus, id, R, 0.1, 1.2, 0.0, 0.2, 10.0, 0.0)

_surrogate_bus_id(fields) = (parse(Int64, fields[1]), String(fields[3]))

function _ggov1_as_tgov1(fields)
    bus, id = _surrogate_bus_id(fields)
    # uqgrid requires Rselect = Fswitch = 1 for the redirect to be meaningful.
    rselect, fswitch = Int(_f(fields, 4)), Int(_f(fields, 5))
    (rselect, fswitch) == (1, 1) ||
        @warn "GGOV1 surrogate at bus $bus id $id has Rselect=$rselect Fswitch=$fswitch (expected 1,1); droop may not be comparable."
    _tgov1_surrogate(bus, id, _f(fields, 6))          # field 6 = R
end

function _hygov_as_tgov1(fields)
    bus, id = _surrogate_bus_id(fields)
    _tgov1_surrogate(bus, id, _f(fields, 4))          # field 4 = permanent droop R
end

function _ieeeg1_as_tgov1(fields)
    bus, id = _surrogate_bus_id(fields)
    K = _f(fields, 6)                                  # field 6 = K = 1/R
    _tgov1_surrogate(bus, id, K > 0 ? 1.0 / K : 0.05)
end

_exciter_as_sexs(fields) = _sexs_surrogate(_surrogate_bus_id(fields)...)

# Tier 1 -- MIRRORED REDIRECTS. uqgrid itself redirects these four models onto
# TGOV1/SEXS rather than implementing them, and we copy its mapping and its
# constants verbatim (uqgrid/uqgrid/io/parse.py). Using them keeps GradPower
# and the reference simulator on the same footing, so a GradPower-vs-uqgrid
# comparison on a case containing them remains meaningful.
const DYR_REDIRECT_MAP = Dict{String,Function}(
    "GGOV1"  => _ggov1_as_tgov1,
    "EXPIC1" => _exciter_as_sexs,
    "SCRX"   => _exciter_as_sexs,
    "ESAC6A" => _exciter_as_sexs,
)

# Tier 2 -- LOCAL SURROGATES. uqgrid implements every one of these NATIVELY, so
# there is no upstream mapping to copy and no oracle behind the substitution.
# They exist only so a case containing them can be run end to end at all; the
# dynamics they produce are NOT the source model's dynamics, and a
# GradPower-vs-uqgrid comparison over them compares different equations.
#
# Do not use Tier 2 for validation, parameter studies, or published results.
# The honest fix is to implement the models (plan_enhance.md phases 2-5);
# ESST4B is the highest-value one at 278 ACTIVSg2000 records.
const DYR_LOCAL_SURROGATE_MAP = Dict{String,Function}(
    "HYGOV"  => _hygov_as_tgov1,
    "IEEEG1" => _ieeeg1_as_tgov1,
    "ESST4B" => _exciter_as_sexs,
    "EXAC1"  => _exciter_as_sexs,
    "EXAC2"  => _exciter_as_sexs,
    "ESAC1A" => _exciter_as_sexs,
    "IEEET1" => _exciter_as_sexs,
    "ESDC2A" => _exciter_as_sexs,
)

const DYR_SURROGATE_MAP = merge(DYR_REDIRECT_MAP, DYR_LOCAL_SURROGATE_MAP)

"""DYR models redirected exactly as uqgrid redirects them."""
redirect_dyr_models() = Set(String.(keys(DYR_REDIRECT_MAP)))

"""DYR models stood in for with no upstream oracle. Not valid for validation."""
local_surrogate_dyr_models() = Set(String.(keys(DYR_LOCAL_SURROGATE_MAP)))

"""Every source DYR model GradPower can stand in for but does not implement."""
surrogate_dyr_models() = Set(String.(keys(DYR_SURROGATE_MAP)))

const _DYR_GOVERNOR_MODELS = Set(["GAST", "GGOV1", "HYGOV", "IEEEG1", "IEESGO", "TGOV1"])
const _DYR_MACHINE_MODELS = Set(["GENROU", "GENSAL"])
const _DYR_EXCITER_MODELS = Set(["ESAC1A", "ESAC6A", "ESDC1A", "ESDC2A", "ESST4B",
                                 "EXAC1", "EXAC2", "EXPIC1", "IEEET1", "SCRX", "SEXS"])
const _DYR_STABILIZER_MODELS = Set(["IEEEST"])
const _DYR_LOAD_MODELS = Set(["CIM5BL"])

struct DyrRecordCoverage
    record_index::Int
    source_model::String
    effective_model::Union{Nothing,String}
    bus::Int64
    device_id::String
    active::Union{Nothing,Bool}
    status::Symbol
end

struct DyrModelCoverage
    total::Int
    active::Int
    inactive::Int
    native::Int
    redirected::Int
    unsupported::Int
    unmatched::Int
    duplicate::Int
end

struct DyrCoverageReport
    raw_path::String
    dyr_path::String
    records::Vector{DyrRecordCoverage}
    by_source_model::Dict{String,DyrModelCoverage}
    active_generators_without_machine::Vector{Tuple{Int64,String}}
end

"""Return the DYR model names implemented by the current `DEVICE_TYPE_MAP`."""
native_dyr_models() = Set(String.(keys(DEVICE_TYPE_MAP)))

function _dyr_model_family(model::String)
    model in _DYR_MACHINE_MODELS && return :machine
    if haskey(DEVICE_TYPE_MAP, model)
        dtype = DEVICE_TYPE_MAP[model]
        dtype <: AbstractGeneratorType && return :machine
        dtype <: AbstractGovernorType && return :governor
        dtype <: AbstractExciterType && return :exciter
        dtype <: AbstractStabilizerType && return :stabilizer
        dtype <: AbstractLoadType && return :load
    end
    model in _DYR_GOVERNOR_MODELS && return :governor
    model in _DYR_EXCITER_MODELS && return :exciter
    model in _DYR_STABILIZER_MODELS && return :stabilizer
    model in _DYR_LOAD_MODELS && return :load
    return Symbol(model)
end

function _coverage_by_source_model(records::Vector{DyrRecordCoverage})
    fields = (:total, :active, :inactive, :native, :redirected,
              :unsupported, :unmatched, :duplicate)
    counters = Dict{String,Dict{Symbol,Int}}()
    for record in records
        counts = get!(counters, record.source_model, Dict(field => 0 for field in fields))
        counts[:total] += 1
        record.active === true && (counts[:active] += 1)
        record.active === false && (counts[:inactive] += 1)
        record.status != :inactive && (counts[record.status] += 1)
    end
    return Dict(model => DyrModelCoverage((counts[field] for field in fields)...)
                for (model, counts) in counters)
end

"""Return aggregate active/static-disposition and classification counts."""
function coverage_counts(report::DyrCoverageReport)
    statuses = (:native, :redirected, :unsupported, :unmatched, :duplicate)
    counts = Dict(status => count(r -> r.status == status, report.records) for status in statuses)
    counts[:active] = count(r -> r.active === true, report.records)
    counts[:inactive] = count(r -> r.active === false, report.records)
    return counts
end

"""Return aggregate coverage for one source model, or all-zero counts if absent."""
function coverage_by_model(report::DyrCoverageReport, model::AbstractString)
    return get(report.by_source_model, uppercase(String(model)), DyrModelCoverage(0, 0, 0, 0, 0, 0, 0, 0))
end

"""Fraction of active DYR records covered by native GradPower models."""
function native_coverage(report::DyrCoverageReport)
    counts = coverage_counts(report)
    return counts[:active] == 0 ? 0.0 : counts[:native] / counts[:active]
end

"""
    analyze_dyr_coverage(raw_path, dyr_path; redirects=Dict())

Classify every DYR record as exactly one of `:native`, `:redirected`,
`:unsupported`, `:unmatched`, `:inactive`, or `:duplicate`. Static equipment is
matched using the external RAW bus number and normalized PSS/E ID. Redirects are
opt-in and never contribute to native coverage.
"""
function analyze_dyr_coverage(raw_path::String, dyr_path::String;
                              redirects::AbstractDict=Dict{String,String}())
    raw = read_psse_raw(raw_path)
    dyr = read_psse_dyr(dyr_path)
    native = native_dyr_models()
    normalized_redirects = Dict(uppercase(String(k)) => uppercase(String(v))
                                for (k, v) in redirects)

    active_generators = Set(_device_key(gen.busn, gen.name) for gen in raw.gens if gen.status == 1)
    inactive_generators = Set(_device_key(gen.busn, gen.name) for gen in raw.gens if gen.status == 0)
    active_loads = Set(_device_key(load.busn, load.name) for load in raw.loads if load.status == 1)
    inactive_loads = Set(_device_key(load.busn, load.name) for load in raw.loads if load.status == 0)

    seen = Set{Tuple{Symbol,Int64,String}}()
    covered_machines = Set{Tuple{Int64,String}}()
    records = DyrRecordCoverage[]
    for (index, values) in enumerate(dyr)
        length(values) >= 3 || throw(ArgumentError("Malformed DYR record $index: expected bus, model, and ID"))
        source_model = uppercase(strip(replace(String(values[2]), "'" => "", "\"" => "")))
        bus = try
            parse(Int64, strip(replace(String(values[1]), "'" => "", "\"" => "")))
        catch
            throw(ArgumentError("Malformed DYR record $index: invalid bus $(repr(values[1]))"))
        end
        key = _device_key(bus, String(values[3]))
        family = _dyr_model_family(source_model)
        identity = (family, key...)

        active_keys, inactive_keys = if family == :load
            active_loads, inactive_loads
        elseif family in (:machine, :governor, :exciter, :stabilizer) ||
               source_model in native || haskey(normalized_redirects, source_model)
            active_generators, inactive_generators
        else
            union(active_generators, active_loads), union(inactive_generators, inactive_loads)
        end

        effective_model = nothing
        status = if identity in seen
            :duplicate
        elseif key in active_keys
            if source_model in native
                effective_model = source_model
                :native
            elseif haskey(normalized_redirects, source_model)
                effective_model = normalized_redirects[source_model]
                :redirected
            else
                :unsupported
            end
        elseif key in inactive_keys
            :inactive
        else
            :unmatched
        end
        push!(seen, identity)
        family == :machine && status in (:native, :redirected) &&
            key in active_generators && push!(covered_machines, key)
        disposition = key in active_keys ? true : key in inactive_keys ? false : nothing
        push!(records, DyrRecordCoverage(index, source_model, effective_model, key[1], key[2],
                                         disposition, status))
    end

    missing_machines = sort!(collect(setdiff(active_generators, covered_machines)))
    return DyrCoverageReport(raw_path, dyr_path, records,
                             _coverage_by_source_model(records), missing_machines)
end

function return_dyr_device(data, dev, ptr)
    ptr += 1
    while dev[end] != "/"
        append!(dev, split(strip(data[ptr]), r"\s*,\s*|\s+"))
        ptr = ptr + 1
    end
    return ptr, dev
end

function read_psse_dyr(dyr_filename)
    devices = []
    data = readlines(dyr_filename)
    ptr = 1
    data_len = length(data)

    while ptr <= data_len
        if occursin(",", data[ptr])
            # Comma delimited file
            dev = split(strip(data[ptr]), r"\s*,\s*")
        else
            dev = split(strip(data[ptr]), r"\s+")
        end

        if length(dev) == 0 || isempty(dev[1])
            # Blank line — split("") yields [""] not [], so check both.
            ptr = ptr + 1
        elseif startswith(dev[1], "//")
            # Comment
            ptr = ptr + 1
        else
            ptr, dev = return_dyr_device(data, dev, ptr)
            push!(devices, dev)
        end
    end
    return devices
end

# ============
# CONSTRUCTORS
# ============

"""
    mat_to_grad(mpc)

Converts a MATPOWER case to a PowerSystem GradPower structure after parsing
with `read_matpower_case`.

Note: Might want to create mpc struct to leverage multiple dispatch and
ensure that the input is a MATPOWER case.
"""
function mat_to_grad(mpc)
    # Initialize empty arrays
    buses = Bus[]
    gens = Gen[]
    loads = Load[]
    branches = Branch[]
    shunts = Shunt[]
    # Initialize an empty busmap
    busmap = Dict{Int64,Int64}()
    baseMVA = mpc["baseMVA"]

    # Iterate over each bus in the input dictionary
    for (index, bus) in enumerate(mpc["bus"])
        # Create a new Bus structure
        new_bus = Bus(bus["bus_i"], string(bus["bus_i"]), bus["type"], bus["baseKV"], bus["Vm"], (π/180.0)*bus["Va"])
        # Append the bus to our buses array
        push!(buses, new_bus)
        # Add a mapping from bus_i to the internal representation
        busmap[bus["bus_i"]] = index
        # If the bus has load, create a Load structure
        if bus["Pd"] > 0.0 || bus["Qd"] > 0.0
            push!(loads, Load(index, " ", bus["Pd"]/baseMVA, -bus["Qd"]/baseMVA))
        end
        # If the bus has shunt, create a Shunt structure
        if bus["Gs"] > 0.0 || bus["Bs"] > 0.0
            push!(shunts, Shunt(index, " ", bus["Gs"]/baseMVA, bus["Bs"]/baseMVA))
        end
    end

    # Convert the rest of the data from the input dictionary
    for gen in mpc["gen"]
        bus = busmap[gen["bus"]]
        status_val = gen["status"]
        @assert status_val == 0.0 || status_val == 1.0 "Gen status must be 0 or 1, got $status_val"
        push!(gens, Gen(bus, " ", gen["Pg"]/baseMVA, gen["Qg"]/baseMVA, gen["mBase"], Bool(status_val),
                        get(gen, "Qmax", Inf)/baseMVA, get(gen, "Qmin", -Inf)/baseMVA))
    end
    for branch in mpc["branch"]
        fr = busmap[branch["fbus"]]
        to = busmap[branch["tbus"]]
        push!(branches, Branch(fr, to, " ", branch["r"], branch["x"], branch["b"], branch["ratio"], branch["angle"]))
    end

    # Construct the PowerSystem structure
    ps = PowerSystem(mpc["baseMVA"], buses, gens, loads, branches, shunts, busmap)
    return ps
end

function raw_to_grad(raw::PsystemRaw)
    # Initialize empty arrays
    buses = Bus[]
    gens = Gen[]
    loads = Load[]
    branches = Branch[]
    shunts = Shunt[]
    # Initialize an empty busmap
    busmap = Dict{Int64,Int64}()
    baseMVA = raw.baseMVA

    for (index, bus) in enumerate(raw.buses)
        new_bus = Bus(bus.busn, bus.name, bus.type, bus.baseKV, bus.vm, (π/180.0)*bus.va)
        push!(buses, new_bus)
        busmap[bus.busn] = index
    end

    for branch in raw.branches
        branch.status == 1 || continue
        fr = busmap[branch.fbus]
        to = busmap[branch.tbus]
        # Note: create constructor that takes r, x, b. No ratio and angle.
        push!(branches, Branch(fr, to, branch.ckt, branch.r, branch.x, branch.b, 0.0, 0.0))
    end

    for tran in raw.transformers
        tran.status == 1 || continue
        fr = busmap[tran.fbus]
        to = busmap[tran.tbus]

        if tran.CW == 2
            @assert false "Transformer control mode 2 not supported"
        else
            volt1 = tran.WINDV1
            volt2 = tran.WINDV2
        end

        # CW==1 with NOMV1 ≠ baseKV(from-bus): rescale impedance by
        # (NOMV1/baseKV_fr)^2 so it lands on the system-base zbase.
        zbase_ratio = 1.0
        if tran.CW == 1 && tran.NOMV1 > 0.0
            zbase_ratio = (tran.NOMV1 / buses[fr].baseKV)^2.0
        end

        if tran.CZ == 1
            r12 = tran.r*(volt2)^2.0*zbase_ratio
            x12 = tran.x*(volt2)^2.0*zbase_ratio
        elseif tran.CZ == 2
            r12 = tran.r*(baseMVA/tran.sbase12)*(volt2)^2.0
            x12 = tran.x*(baseMVA/tran.sbase12)*(volt2)^2.0
        elseif tran.CZ == 3
            @assert false "Not implemented yet"
        end

        tap = volt1/volt2
        # MAG2 enters as the branch's line-charging susceptance (π-equivalent
        # convention puts MAG2/2 on each end).
        mag_sh = abs(tran.MAG2) > 0.0 ? tran.MAG2 : 0.0
        push!(branches, Branch(fr, to, tran.ckt, r12, x12, mag_sh, tap, tran.ANG1))

        if tran.COD1 == 1
            push!(shunts, Shunt(fr, "tran", tran.MAG1*baseMVA, tran.MAG2*baseMVA))
        end
    end

    # Three-winding transformers — star-point decomposition.
    # Adds a synthetic dummy bus at the star point + three two-winding sub-branches.
    # Per-winding service status from the encoded status field (0–4).
    if !isempty(raw.transthree)
        max_busn = isempty(busmap) ? 0 : maximum(keys(busmap))
        kdummy = 0
        for tran in raw.transthree
            ibus = busmap[tran.ibus]
            jbus = busmap[tran.jbus]
            kbus = busmap[tran.kbus]

            # add dummy star-point bus with initial vmstar / anstar
            star_busn = max_busn + 1 + kdummy
            star_idx = length(buses) + 1
            push!(buses, Bus(star_busn, "STAR", 1, 0.0, tran.vmstar, (π/180.0)*tran.anstar))
            busmap[star_busn] = star_idx
            kdummy += 1

            if tran.CW == 2
                baseKV1 = buses[ibus].baseKV
                baseKV2 = buses[jbus].baseKV
                baseKV3 = buses[kbus].baseKV
                volt1 = tran.WINDV1/baseKV1
                volt2 = tran.WINDV2/baseKV2
                volt3 = tran.WINDV3/baseKV3
            else
                volt1 = tran.WINDV1
                volt2 = tran.WINDV2
                volt3 = tran.WINDV3
            end

            if tran.CZ == 1
                r12, x12 = tran.r12, tran.x12
                r23, x23 = tran.r23, tran.x23
                r13, x13 = tran.r13, tran.x13
            elseif tran.CZ == 2
                r12 = tran.r12 * (baseMVA/tran.sbase12)
                x12 = tran.x12 * (baseMVA/tran.sbase12)
                r23 = tran.r23 * (baseMVA/tran.sbase23)
                x23 = tran.x23 * (baseMVA/tran.sbase23)
                r13 = tran.r13 * (baseMVA/tran.sbase31)
                x13 = tran.x13 * (baseMVA/tran.sbase31)
            else  # CZ == 3 (load-loss watts + Z in pu)
                r12 = (tran.r12 / 1e6) / tran.sbase12
                r23 = (tran.r23 / 1e6) / tran.sbase23
                r13 = (tran.r13 / 1e6) / tran.sbase31
                x12 = sqrt(tran.x12^2 - r12^2)
                x23 = sqrt(tran.x23^2 - r23^2)
                x13 = sqrt(tran.x13^2 - r13^2)
                r12 *= baseMVA/tran.sbase12; x12 *= baseMVA/tran.sbase12
                r23 *= baseMVA/tran.sbase23; x23 *= baseMVA/tran.sbase23
                r13 *= baseMVA/tran.sbase31; x13 *= baseMVA/tran.sbase31
            end

            r1 = 0.5*(r12 + r13 - r23); x1 = 0.5*(x12 + x13 - x23)
            r2 = 0.5*(r12 - r13 + r23); x2 = 0.5*(x12 - x13 + x23)
            r3 = 0.5*(r13 + r23 - r12); x3 = 0.5*(x13 + x23 - x12)

            # status code: 1 -> all in service, 2 -> wind2 out, 3 -> wind3 out, 4 -> wind1 out, 0 -> all out
            s1, s2, s3 = if tran.status == 1
                (true, true, true)
            elseif tran.status == 2
                (true, false, true)
            elseif tran.status == 3
                (true, true, false)
            elseif tran.status == 4
                (false, true, true)
            else
                (false, false, false)
            end

            s1 && push!(branches, Branch(ibus, star_idx, tran.ckt, r1, x1, 0.0, volt1, tran.ANG1))
            s2 && push!(branches, Branch(star_idx, jbus, tran.ckt, r2, x2, 0.0, volt2, tran.ANG2))
            s3 && push!(branches, Branch(star_idx, kbus, tran.ckt, r3, x3, 0.0, volt3, tran.ANG3))
        end
    end

    # First pass: track which buses still have an active generator.
    buses_with_active_gen = Set{Int}()
    for gen in raw.gens
        gen.status == 1 || continue
        push!(buses_with_active_gen, busmap[gen.busn])
    end

    for gen in raw.gens
        gen.status == 1 || continue
        bus = busmap[gen.busn]
        push!(gens, Gen(bus, gen.name, gen.pg/baseMVA, gen.qg/baseMVA, gen.mbase, gen.status,
                        gen.qt/baseMVA, gen.qb/baseMVA))
        # PV/SLACK buses: voltage setpoint comes from the generator's vs field,
        # not the bus's flat-start magnitude.
        bt = buses[bus].type
        if bt == 2 || bt == 3
            buses[bus].v0m = gen.vs
        end
    end

    # Downgrade PV (type=2) buses to PQ (type=1) when no active gen remains.
    # SLACK (type=3) stays as-is by convention.
    for (idx, bus) in enumerate(buses)
        if bus.type == 2 && !(idx in buses_with_active_gen)
            bus.type = 1
        end
    end

    for load in raw.loads
        load.status == 1 || continue
        bus = busmap[load.busn]
        push!(loads, Load(bus, load.name, load.pl/baseMVA, -load.ql/baseMVA))
    end

    for shunt in raw.shunts
        shunt.status == 1 || continue
        bus = busmap[shunt.busn]
        push!(shunts, Shunt(bus, shunt.name, shunt.gshunt/baseMVA, shunt.bshunt/baseMVA))
    end

    for sshunt in raw.switched_shunts
        if sshunt.status == 1
            bus = busmap[sshunt.busn]
            push!(shunts, Shunt(bus, "swsh", 0.0, sshunt.binit/baseMVA))
        end
    end

    # Construct the PowerSystem structure
    ps = PowerSystem(raw.baseMVA, buses, gens, loads, branches, shunts, busmap)
    return ps
end

"""
    create_device_vector(devices)

Converts a vector of PSSE dyr devices to a vector of AbstractDeviceType structs.
"""
function create_device_vector(devices;
                               active_gen_keys::Union{Nothing,Set{Tuple{Int64,String}}}=nothing,
                               surrogates::Bool=false)
    psse_devices = Vector{GradPower.AbstractDeviceType}()
    skipped_inactive_gen = 0
    skipped_orphan_ctrl = 0
    unknown_types = String[]
    redirected_types = String[]
    kept_gen_keys = Set{Tuple{Int64,String}}()

    parsed = Tuple{GradPower.AbstractDeviceType,String}[]
    for device in devices
        device_type_name = strip(device[2], ''')  # strip apostrophes
        if haskey(DEVICE_TYPE_MAP, device_type_name)
            device_type = DEVICE_TYPE_MAP[device_type_name]
            dev = from_data_fields(device_type, device)
            push!(parsed, (dev, String(device_type_name)))
        elseif surrogates && haskey(DYR_SURROGATE_MAP, device_type_name)
            # Stand-in, not an implementation -- tracked separately from native.
            dev = DYR_SURROGATE_MAP[device_type_name](device)
            push!(redirected_types, String(device_type_name))
            push!(parsed, (dev, String(device_type_name)))
        else
            push!(unknown_types, String(device_type_name))
        end
    end

    # Pass 1: keep generators whose (bus, id) is an active static gen.
    for (dev, _) in parsed
        dev isa AbstractGeneratorType || continue
        key = (dev.bus, _normalize_id(dev.id))
        if active_gen_keys === nothing || key in active_gen_keys
            push!(kept_gen_keys, key)
            push!(psse_devices, dev)
        else
            skipped_inactive_gen += 1
        end
    end

    # Pass 2: keep controllers (governor, exciter) only if their target Genrou
    # at the same (bus, id) survived pass 1. Otherwise the controller would
    # wire to nothing and produce silent NaNs.
    kept_exc_keys = Set{Tuple{Int64,String}}()
    for (dev, _) in parsed
        dev isa AbstractGeneratorType && continue
        if dev isa AbstractGenControlType
            key = (dev.bus, _normalize_id(dev.id))
            if !(key in kept_gen_keys)
                skipped_orphan_ctrl += 1
                continue
            end
            if dev isa AbstractStabilizerType
                # defer stabilizers to pass 3
                continue
            end
            if dev isa AbstractExciterType
                kept_exc_keys = push!(kept_exc_keys, key)
            end
        end
        push!(psse_devices, dev)
    end

    # Pass 3: keep stabilizers only if their target exciter was kept.
    for (dev, _) in parsed
        dev isa AbstractStabilizerType || continue
        key = (dev.bus, _normalize_id(dev.id))
        if key in kept_gen_keys && key in kept_exc_keys
            push!(psse_devices, dev)
        else
            skipped_orphan_ctrl += 1
        end
    end

    if skipped_inactive_gen > 0
        @info "Skipped $skipped_inactive_gen dynamic generator row(s) with no matching active static gen."
    end
    if skipped_orphan_ctrl > 0
        @info "Skipped $skipped_orphan_ctrl controller row(s) whose target generator was filtered."
    end
    if !isempty(redirected_types)
        mirrored = Dict{String,Int}()
        local_sub = Dict{String,Int}()
        for t in redirected_types
            d = haskey(DYR_REDIRECT_MAP, t) ? mirrored : local_sub
            d[t] = get(d, t, 0) + 1
        end
        if !isempty(mirrored)
            @info "Applied uqgrid-mirrored DYR redirects to $(sum(values(mirrored))) record(s): $mirrored"
        end
        if !isempty(local_sub)
            @warn """LOCAL SURROGATES applied to $(sum(values(local_sub))) .dyr record(s): $local_sub
                     uqgrid implements each of these natively; GradPower does not, and these
                     stand-ins reproduce TGOV1/SEXS dynamics, NOT the source equations. A
                     GradPower-vs-uqgrid comparison over these records compares DIFFERENT MODELS.
                     Valid only to get a case running end to end -- never for validation,
                     parameter studies, or published results. See DYR_LOCAL_SURROGATE_MAP."""
        end
    end
    if !isempty(unknown_types)
        counts = Dict{String,Int}()
        for t in unknown_types
            counts[t] = get(counts, t, 0) + 1
        end
        @warn "Unknown device type(s) in .dyr (skipped): $counts"
    end

    return psse_devices
end
