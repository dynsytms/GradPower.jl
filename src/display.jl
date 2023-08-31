function display_parameters(sys::PowerSystem)
    # iterate all the devices and display parameters.
    println("Displaying parameters...")

    if sys.dynamic == nothing
        println("No dynamic data found.")
        return
    end

    for (i, device) in enumerate(sys.dynamic.devices)
        ptr = device.par_ptr
        dtype = device.dtype
        @printf("Device %d: %s\n", i, get_device_name(dtype))
        for (j, par) in enumerate(get_param_names(dtype))
            @printf("  [%d] %s\n", ptr + (j - 1), par)
        end
    end
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

    for gen in sys.gens
        psch_str = @sprintf("%.3e", gen.psch)
        qsch_str = @sprintf("%.3e", gen.qsch)
        println(io, "$(gen.bus)\t$(gen.id)\t\t$(psch_str)\t$(qsch_str)\t$(gen.mbase)")
    end

    println(io, "\nLoads:")
    println(io, "===================================")
    println(io, "bus\tid\t\tpd\tqd")
    println(io, "-----------------------------------")
    for load in sys.loads
        println(io, "$(load.bus)\t$(load.id)\t\t$(load.pd)\t$(load.qd)")
    end

    println(io, "\nBranches:")
    println(io, "=============================================================================================")
    println(io, "fr\tto\tid\t\tr\tx\tsh\ttap\tshift")
    println(io, "---------------------------------------------------------------------------------------------")
    for branch in sys.branches
        println(io, "$(branch.fr)\t$(branch.to)\t$(branch.id)\t\t$(branch.r)\t$(branch.x)\t$(branch.sh)\t$(branch.tap)\t$(branch.shift)")
    end

    println(io, "\nShunts:")
    println(io, "==============================")
    println(io, "bus\tid\t\tgsh\tbsh")
    println(io, "------------------------------")
    for shunt in sys.shunts
        println(io, "$(shunt.bus)\t$(shunt.id)\t\t$(shunt.gsh)\t$(shunt.bsh)")
    end
end
