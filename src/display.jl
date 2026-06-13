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

function get_index_info(idx::Int, ps::PowerSystem)
    # given an index from the state vector, find out the state variable
    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    ctrl_dim = ps.dynamic.ctrl_dim
    par_dim = ps.dynamic.par_dim
    nbus = length(ps.buses)

    if idx > diff_dim + alg_dim + 2*nbus
        println("Index out of bounds.")
    elseif idx <= diff_dim
        for i in 1:length(ps.dynamic.devices) - 1
            device = ps.dynamic.devices[i]
            device2 = ps.dynamic.devices[i + 1]

            diff_ptr = device.diff_ptr
            diff_ptr2 = device2.diff_ptr
            if diff_ptr <= idx < diff_ptr2
                println("Device: ", i)
                println("State variable: ", get_diff_names(device.dtype)[idx - diff_ptr + 1])
                show(ps.dynamic.devices[i].dtype)
                return
            end
        end
    elseif idx <= diff_dim + alg_dima
        idx = idx - diff_dim
        for i in 1:length(ps.dynamic.devices) - 1
            device = ps.dynamic.devices[i]
            device2 = ps.dynamic.devices[i + 1]

            alg_ptr = device.alg_ptr
            alg_ptr2 = device2.alg_ptr
            if alg_ptr <= idx < alg_ptr2
                println("Device: ", i)
                println("State variable: ", get_alg_names(device.dtype)[idx - alg_ptr + 1])
                return
            end
        end
    else idx <= diff_dim + alg_dim + 2*nbus
        idx = idx - diff_dim - alg_dim
        if idx % 2 == 0
            println("Bus: ", idx/2)
            println("State variable: ", "v")
            return
        else
            println("Bus: ", (idx + 1)/2)
            println("State variable: ", "δ")
            return
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

function show(io::IO, bus::Bus)
    println(io, "Bus:")
    println(io, "  i: ", bus.i)
    println(io, "  id: ", bus.id)
    println(io, "  type: ", bus.type)
    println(io, "  baseKV: ", bus.baseKV)
    println(io, "  v0m: ", bus.v0m)
    println(io, "  v0a: ", bus.v0a)
end

function find_outliers(vec::Vector{Float64})
    # finds outliers in a vector of Float64
    # returns indices of outliers
    mean_val = mean(vec)
    std_val = std(vec)
    outliers = findall(x -> x > mean_val + 2.5 * std_val || x < mean_val - 3.5 * std_val, vec)
    return outliers
end

function get_bus_info(busn::Int64, sys::PowerSystem)
    # prints all information about a bus. including connected branches, connected generators,
    # connected loads, etc.

    # find the bus
    bus_idx = findfirst(x -> x.i == busn, sys.buses)

    if bus_idx == nothing
        println("Bus not found.")
        return
    end
    show(stdout, sys.buses[bus_idx])

    # find connected generators
    gen_idx = findall(x -> x.bus == bus_idx, sys.gens)
    if length(gen_idx) > 0
        println("\nConnected Generators:")
        println("================================================================")
        println("bus\tid\t\tpsch\t\tqsch\t\tmbase")
        println("----------------------------------------------------------------")
        for idx in gen_idx
            gen = sys.gens[idx]
            psch_str = @sprintf("%.3e", gen.psch)
            qsch_str = @sprintf("%.3e", gen.qsch)
            println("$(gen.bus)\t$(gen.id)\t\t$(psch_str)\t$(qsch_str)\t$(gen.mbase)")

            # print dynamic model


        end
    end

    # find connected loads
    load_idx = findall(x -> x.bus == bus_idx, sys.loads)
    if length(load_idx) > 0
        println("\nConnected Loads:")
        println("===================================")
        println("bus\tid\t\tpd\tqd")
        println("-----------------------------------")
        for idx in load_idx
            load = sys.loads[idx]
            println("$(load.bus)\t$(load.id)\t\t$(load.pd)\t$(load.qd)")
        end
    end
end
