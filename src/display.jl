"""
    display_parameters([io], ps)

Print device parameter names and their current vector indices to `io` (stdout).
"""
function display_parameters(io::IO, sys::PowerSystem)
    println(io, "Displaying parameters...")

    if sys.dynamic == nothing
        println(io, "No dynamic data found.")
        return
    end

    for (i, device) in enumerate(sys.dynamic.devices)
        ptr = device.par_ptr
        dtype = device.dtype
        @printf(io, "Device %d: %s\n", i, get_device_name(dtype))
        for (j, par) in enumerate(get_param_names(dtype))
            @printf(io, "  [%d] %s\n", ptr + (j - 1), par)
        end
    end
end
display_parameters(sys::PowerSystem) = display_parameters(stdout, sys)

"""
    get_index_info(idx, ps) -> StateChannel
    get_index_info(io, idx, ps) -> StateChannel

Return metadata for a current raw state index, raising `BoundsError` outside
the state vector. The IO overload also prints the channel.
"""
function get_index_info(idx::Integer, ps::PowerSystem)
    return state_channels(ps)[idx]
end

function get_index_info(io::IO, idx::Integer, ps::PowerSystem)
    channel = get_index_info(idx, ps)
    println(io, "State [", channel.index, "]: ", channel.model, " bus=", channel.bus,
            " id=", repr(channel.device_id), " ", channel.name, " (", channel.kind, ")")
    return channel
end

function show(io::IO, sys::PowerSystem)
    println(io, "PowerSystem:")
    println(io, "  baseMVA: ", sys.baseMVA)

    println(io, "\nBuses:")
    println(io, "================================================================")
    println(io, "i\tid\t\t\ttype\tbaseKV\tv0m\tv0a")
    println(io, "----------------------------------------------------------------")
    for bus in sys.buses
        v0a_str = @sprintf("%.3e", bus.v0a)
        println(io, "$(bus.i)\t$(bus.id)\t\t$(bus.type)\t$(bus.baseKV)\t$(bus.v0m)\t$(v0a_str)")
    end

    println(io, "\nGens:")
    println(io, "===============================================================")
    println(io, "bus\tid\t\tpsch\t\tqsch\t\tmbase")
    println(io, "---------------------------------------------------------------")

    for gen in system_summary(sys; section=:generators)
        psch_str = @sprintf("%.3e", gen.psch)
        qsch_str = @sprintf("%.3e", gen.qsch)
        println(io, "$(gen.bus)\t$(gen.id)\t\t$(psch_str)\t$(qsch_str)\t$(gen.mbase)")
    end

    println(io, "\nLoads:")
    println(io, "===================================")
    println(io, "bus\tid\t\tpd\tqd")
    println(io, "-----------------------------------")
    for load in system_summary(sys; section=:loads)
        println(io, "$(load.bus)\t$(load.id)\t\t$(load.pd)\t$(load.qd)")
    end

    println(io, "\nBranches:")
    println(io, "=============================================================================================")
    println(io, "fr\tto\tid\t\tr\tx\tsh\ttap\tshift")
    println(io, "---------------------------------------------------------------------------------------------")
    for branch in sys.branches
        println(io, "$(sys.buses[branch.fr].i)\t$(sys.buses[branch.to].i)\t$(branch.id)\t\t$(branch.r)\t$(branch.x)\t$(branch.sh)\t$(branch.tap)\t$(branch.shift)")
    end

    println(io, "\nShunts:")
    println(io, "==============================")
    println(io, "bus\tid\t\tgsh\tbsh")
    println(io, "------------------------------")
    for shunt in sys.shunts
        println(io, "$(sys.buses[shunt.bus].i)\t$(shunt.id)\t\t$(shunt.gsh)\t$(shunt.bsh)")
    end
end

function show(io::IO, bus::Bus)
    println(io, "Bus:")
    println(io, "  i: ", bus.i)
    println(io, "  id: ", bus.id)
    println(io, "  type: ", bus.type)
    println(io, "  baseKV: ", bus.baseKV)
    println(io, "  v0m: ", bus.v0m)
    println(io, "  v0a: ", bus.v0a)
end

"""
    get_bus_info([io], bus, ps)

Print a bus and its connected static rows, branches, shunts, and dynamic devices.
All displayed bus numbers are external; output defaults to stdout.
"""
function get_bus_info(io::IO, busn::Integer, sys::PowerSystem)
    b = bus_index(sys, busn)
    show(io, sys.buses[b])
    for (section, heading) in ((:generators, "Generators"), (:loads, "Loads"),
                               (:devices, "Devices"))
        rows = filter(r -> r.bus == busn, system_summary(sys; section))
        isempty(rows) && continue
        println(io, "\nConnected ", heading, ":")
        for row in rows
            println(io, row)
        end
    end
    branches = filter(r -> r.fr == b || r.to == b, sys.branches)
    if !isempty(branches)
        println(io, "\nConnected Branches:")
        for branch in branches
            println(io, (fr=sys.buses[branch.fr].i, to=sys.buses[branch.to].i,
                         id=_normalize_ckt(branch.id), r=branch.r, x=branch.x,
                         sh=branch.sh, tap=branch.tap, shift=branch.shift))
        end
    end
    shunts = filter(r -> r.bus == b, sys.shunts)
    if !isempty(shunts)
        println(io, "\nConnected Shunts:")
        for shunt in shunts
            println(io, (bus=busn, id=_normalize_id(shunt.id), gsh=shunt.gsh, bsh=shunt.bsh))
        end
    end
    return nothing
end
get_bus_info(busn::Integer, sys::PowerSystem) = get_bus_info(stdout, busn, sys)
