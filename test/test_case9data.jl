function create_case9()
    # Initialize empty arrays
    buses = Bus[]
    gens = Gen[]
    loads = Load[]
    branches = Branch[]
    shunts = Shunt[]

    # Initialize an empty busmap
    busmap = Dict{Int64,Int64}()
    baseMVA = 100.0

    # Create buses
    push!(buses, Bus(1, "1", 3, 345.0, 1.0, 0.0))
    push!(buses, Bus(2, "2", 2, 345.0, 1.0, 0.0))
    push!(buses, Bus(3, "3", 2, 345.0, 1.0, 0.0))
    push!(buses, Bus(4, "4", 1, 345.0, 1.0, 0.0))
    push!(buses, Bus(5, "5", 1, 345.0, 1.0, 0.0))
    push!(buses, Bus(6, "6", 1, 345.0, 1.0, 0.0))
    push!(buses, Bus(7, "7", 1, 345.0, 1.0, 0.0))
    push!(buses, Bus(8, "8", 1, 345.0, 1.0, 0.0))
    push!(buses, Bus(9, "9", 1, 345.0, 1.0, 0.0))

    # Create generators
    push!(gens, Gen(1, "1", 0.0, 0.0, 100.0, 1))
    push!(gens, Gen(2, "1", 1.63, 0.0, 100.0, 1))
    push!(gens, Gen(3, "1", 0.85, 0.0, 100.0, 1))

    # Create loads
    push!(loads, Load(5, "1", 0.9, -0.3))
    push!(loads, Load(7, "1", 1.0, -0.35))
    push!(loads, Load(9, "1", 1.25, -0.5))

    # Create branches
    push!(branches, Branch(1, 4, "1", 0.0, 0.0576, 0.0, 0.0, 0.0))
    push!(branches, Branch(4, 5, "1", 0.017, 0.092, 0.158, 0.0, 0.0))
    push!(branches, Branch(5, 6, "1", 0.039, 0.17, 0.358, 0.0, 0.0))
    push!(branches, Branch(3, 6, "1", 0.0, 0.0586, 0.0, 0.0, 0.0))
    push!(branches, Branch(6, 7, "1", 0.0119, 0.1008, 0.209, 0.0, 0.0))
    push!(branches, Branch(7, 8, "1", 0.0085, 0.072, 0.149, 0.0, 0.0))
    push!(branches, Branch(8, 2, "1", 0.0, 0.0625, 0.0, 0.0, 0.0))
    push!(branches, Branch(8, 9, "1", 0.032, 0.161, 0.306, 0.0, 0.0))
    push!(branches, Branch(9, 4, "1", 0.01, 0.085, 0.176, 0.0, 0.0))

    # Create busmap
    for (index, bus) in enumerate(buses)
        busmap[bus.i] = index
    end

    # Construct the PowerSystem structure
    ps = PowerSystem(baseMVA, buses, gens, loads, branches, shunts, busmap)
    return ps
end
