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
function from_psse(raw_file::String, dyr_file::Union{String, Nothing})
    raw = read_psse_raw(raw_file)
    sys = raw_to_grad(raw)
    if dyr_file !== nothing
        psd = PowerSystemDynamics(dyr_file)
        set_dynamics!(sys, psd)
    end
    return sys
end

# ======================
#  PSSE Dynamics (*.dyr)
# ======================

const DEVICE_TYPE_MAP = Dict(
    "GENROU" => Genrou,
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

        if length(dev) == 0
            # Empty
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
        push!(gens, Gen(bus, " ", gen["Pg"]/baseMVA, gen["Qg"]/baseMVA, gen["mBase"]))
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
        fr = busmap[branch.fbus]
        to = busmap[branch.tbus]
        # Note: create constructor that takes r, x, b. No ratio and angle.
        push!(branches, Branch(fr, to, branch.ckt, branch.r, branch.x, branch.b, 0.0, 0.0))
    end

    for tran in raw.transformers
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
        push!(branches, Branch(fr, to, tran.ckt, r12, x12, 0.0, tap, tran.ANG1))

        if tran.COD1 == 1
            push!(shunts, Shunt(fr, "tran", tran.MAG1*baseMVA, tran.MAG2*baseMVA))
        end
    end

    @assert length(raw.transthree) == 0 "Three-winding transformers not supported. yet."

    for gen in raw.gens
        bus = busmap[gen.busn]
        push!(gens, Gen(bus, gen.name, gen.pg/baseMVA, gen.qg/baseMVA, gen.mbase))
    end

    for load in raw.loads
        bus = busmap[load.busn]
        push!(loads, Load(bus, load.name, load.pl/baseMVA, -load.ql/baseMVA))
    end

    for shunt in raw.shunts
        bus = busmap[shunt.busn]
        push!(shunts, Shunt(bus, shunt.name, shunt.gshunt/baseMVA, shunt.bshunt/baseMVA))
    end

    # Construct the PowerSystem structure
    ps = PowerSystem(raw.baseMVA, buses, gens, loads, branches, shunts, busmap)
    return ps
end

"""
    create_device_vector(devices)

Converts a vector of PSSE dyr devices to a vector of AbstractDeviceType structs.
"""
function create_device_vector(devices)
    psse_devices = Vector{GradPower.AbstractDeviceType}()

    for device in devices
        device_type_name = strip(device[2], ''')  # strip apostrophes
        if haskey(DEVICE_TYPE_MAP, device_type_name)
            device_type = DEVICE_TYPE_MAP[device_type_name]
            push!(psse_devices, from_data_fields(device_type, device))
        else
            @warn "Unknown device type: $device_type_name"
        end
    end

    return psse_devices
end
