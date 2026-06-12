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
                    add_static_gen_stubs::Bool=true)
    raw = read_psse_raw(raw_file)
    sys = raw_to_grad(raw)
    if dyr_file !== nothing
        # Mirror uqgrid (io/parse.py:407): drop GENROU/GENSAL dynamic rows
        # whose (bus, id) doesn't reference an active static generator.
        # The raw_to_grad pass has already filtered status==0 gens, so the
        # static `sys.gens` is exactly the active set.
        active = Set{Tuple{Int64,String}}()
        for gen in sys.gens
            push!(active, (sys.buses[gen.bus].i, _normalize_id(gen.id)))
        end
        psd = PowerSystemDynamics(dyr_file; active_gen_keys=active)
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
    # add more device types here
)

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
        push!(gens, Gen(bus, " ", gen["Pg"]/baseMVA, gen["Qg"]/baseMVA, gen["mBase"], Bool(status_val)))
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

        if tran.CZ == 1
            r12 = tran.r*(volt2)^2.0
            x12 = tran.x*(volt2)^2.0
        elseif tran.CZ == 2
            r12 = tran.r*(baseMVA/tran.sbase12)*(volt2)^2.0
            x12 = tran.x*(baseMVA/tran.sbase12)*(volt2)^2.0
        elseif tran.CZ == 3
            @assert false "Not implemented yet"
        end

        tap = volt1/volt2
        # MAG2 enters as the branch's line-charging susceptance (π-equivalent
        # convention puts MAG2/2 on each end). Mirrors uqgrid io/parse.py:70.
        mag_sh = abs(tran.MAG2) > 0.0 ? tran.MAG2 : 0.0
        push!(branches, Branch(fr, to, tran.ckt, r12, x12, mag_sh, tap, tran.ANG1))

        if tran.COD1 == 1
            push!(shunts, Shunt(fr, "tran", tran.MAG1*baseMVA, tran.MAG2*baseMVA))
        end
    end

    # Three-winding transformers — star-point decomposition (per uqgrid io/parse.py:82).
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
        push!(gens, Gen(bus, gen.name, gen.pg/baseMVA, gen.qg/baseMVA, gen.mbase, gen.status))
        # PV/SLACK buses: voltage setpoint comes from the generator's vs field,
        # not the bus's flat-start magnitude. Mirrors uqgrid io/parse.py:189.
        bt = buses[bus].type
        if bt == 2 || bt == 3
            buses[bus].v0m = gen.vs
        end
    end

    # Downgrade PV (type=2) buses to PQ (type=1) when no active gen remains —
    # mirrors uqgrid io/parse.py:195. SLACK (type=3) stays as-is by convention.
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
                               active_gen_keys::Union{Nothing,Set{Tuple{Int64,String}}}=nothing)
    psse_devices = Vector{GradPower.AbstractDeviceType}()
    skipped_inactive_gen = 0
    skipped_orphan_ctrl = 0
    unknown_types = String[]
    kept_gen_keys = Set{Tuple{Int64,String}}()

    parsed = Tuple{GradPower.AbstractDeviceType,String}[]
    for device in devices
        device_type_name = strip(device[2], ''')  # strip apostrophes
        if haskey(DEVICE_TYPE_MAP, device_type_name)
            device_type = DEVICE_TYPE_MAP[device_type_name]
            dev = from_data_fields(device_type, device)
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
    for (dev, _) in parsed
        dev isa AbstractGeneratorType && continue
        if dev isa AbstractGenControlType
            key = (dev.bus, _normalize_id(dev.id))
            if !(key in kept_gen_keys)
                skipped_orphan_ctrl += 1
                continue
            end
        end
        push!(psse_devices, dev)
    end

    if skipped_inactive_gen > 0
        @info "Skipped $skipped_inactive_gen dynamic generator row(s) with no matching active static gen."
    end
    if skipped_orphan_ctrl > 0
        @info "Skipped $skipped_orphan_ctrl controller row(s) whose target generator was filtered."
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
