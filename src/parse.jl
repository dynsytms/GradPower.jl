using Match
using Printf

function parse_line(line, field_names)
    split_line = split(line)
    parsed_line = map(x -> tryparse(Float64, x), split_line)
    return Dict(zip(field_names, parsed_line))
end

function parse_data(block_data, field_names)
    lines = split(block_data, '\n')
    lines = filter(line -> !isempty(strip(line)), lines)  # remove empty lines
    data = map(line -> parse_line(line, field_names), lines)
    return data
end

#   parse_case(file_name)
#
#   Parses a MATPOWER case file and returns a dictionary.
function parse_case(file_name)
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
            mpc[block] = parse_data(block_data, fields)
        end
    end

    return mpc
end

function mat_to_grad(mpc)
    # Initialize empty arrays
    buses = Bus[]
    gens = Gen[]
    loads = Load[]
    branches = Branch[]
    shunts = Shunt[]
    # Initialize an empty busmap
    busmap = Dict{Int64,Int64}()

    # Iterate over each bus in the input dictionary
    for (index, bus) in enumerate(mpc["bus"])
        # Create a new Bus structure
        new_bus = Bus(bus["bus_i"], string(bus["bus_i"]), bus["type"], bus["baseKV"], bus["Vm"], bus["Va"])
        # Append the bus to our buses array
        push!(buses, new_bus)
        # Add a mapping from bus_i to the internal representation
        busmap[bus["bus_i"]] = index
        # If the bus has load, create a Load structure
        if bus["Pd"] > 0.0 || bus["Qd"] > 0.0
            push!(loads, Load(index, bus["Pd"], bus["Qd"]))
        end
        # If the bus has shunt, create a Shunt structure
        if bus["Gs"] > 0.0 || bus["Bs"] > 0.0
            push!(shunts, Shunt(index, bus["Gs"], bus["Bs"]))
        end
    end

    # Convert the rest of the data from the input dictionary
    for gen in mpc["gen"]
        bus = busmap[gen["bus"]]
        push!(gens, Gen(bus, gen["Pg"], gen["Qg"], gen["mBase"]))
    end
    for branch in mpc["branch"]
        fr = busmap[branch["fbus"]]
        to = busmap[branch["tbus"]]
        push!(branches, Branch(fr, to, branch["r"], branch["x"], branch["b"], branch["ratio"], branch["angle"]))
    end

    # Construct the PowerSystem structure
    ps = PowerSystem(mpc["baseMVA"], buses, gens, loads, branches, shunts, busmap)
    return ps
end
