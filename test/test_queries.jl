using Test
using GradPower
using GradPower: bus_index, get_bus, find_generator, find_load, find_devices,
                 find_device, state_index, parameter_index, system_summary,
                 get_index_info, display_parameters, get_bus_info

function _query_fixture(; dynamics=true, reverse_machines=false)
    buses = [Bus(101, "MACHINES", 3, 230.0, 1.0, 0.0),
             Bus(7, "LOAD", 1, 230.0, 1.0, 0.0),
             Bus(305, "STATIC", 2, 230.0, 1.0, 0.0)]
    gens = [Gen(1, "'A '", 0.4, 0.1, 100.0, true),
            Gen(1, "'B '", 0.3, 0.1, 100.0, true),
            Gen(3, "'C '", 0.1, 0.0, 100.0, true),
            Gen(3, "'D '", 0.1, 0.0, 100.0, true)]
    loads = [Load(2, "'L '", 0.8, 0.2)]
    branches = [Branch(1, 2, "'1 '", 0.01, 0.1, 0.0, 0.0, 0.0)]
    shunts = [Shunt(2, "'S '", 0.0, 0.01)]
    ps = PowerSystem(100.0, buses, gens, loads, branches, shunts,
                     Dict(b.i => i for (i, b) in enumerate(buses)))
    if dynamics
        dyn = PowerSystemDynamics()
        ids = reverse_machines ? ("A", "B") : ("B", "A")
        add_device!(dyn, GradPower.GenericGenerator(101, "'$(ids[1]) '"))
        add_device!(dyn, GradPower.TGOV1(101, "'A '", 0.05, 1.0, 2.0, 0.0,
                                       0.5, 1.0, 0.0))
        add_device!(dyn, GradPower.GenericGenerator(101, "'$(ids[2]) '"))
        add_device!(dyn, GradPower.TGOV1(101, "'B '", 0.05, 1.0, 2.0, 0.0,
                                       0.5, 1.0, 0.0))
        set_dynamics!(ps, dyn)
    end
    return ps
end

@testset "Explicit bus numbering and strict static lookups" begin
    ps = _query_fixture()
    @test bus_index(ps, 101) == 1
    @test bus_index(ps, 7) == 2
    @test bus_index(ps, 2; numbering=:internal) == 2
    @test get_bus(ps, 305) === ps.buses[3]
    @test get_bus(ps, 1; numbering=:internal) === ps.buses[1]
    @test_throws ArgumentError bus_index(ps, 1)
    @test_throws ArgumentError bus_index(ps, 0; numbering=:internal)
    @test_throws ArgumentError bus_index(ps, 4; numbering=:internal)
    for mode in (:auto, :EXTERNAL, "external", nothing)
        @test_throws ArgumentError bus_index(ps, 101; numbering=mode)
        @test_throws ArgumentError find_devices(ps; numbering=mode)
    end
    @test find_generator(ps, 101, " A ") == 1
    @test find_generator(ps, 1, "'B'"; numbering=:internal) == 2
    @test find_load(ps, 7, " 'L' ") == 1
    @test find_load(ps, 2, "L"; numbering=:internal) == 1
    @test_throws ArgumentError find_generator(ps, 101, "missing")
    @test_throws ArgumentError find_load(ps, 101, "L")
    @test GradPower.find_gen(ps, Int64(1), "A") == 1
    push!(ps.gens, deepcopy(ps.gens[1]))
    push!(ps.loads, deepcopy(ps.loads[1]))
    @test_throws ArgumentError find_generator(ps, 101, "A")
    @test_throws ArgumentError find_load(ps, 7, "L")
    @test GradPower.find_gen(ps, Int64(1), "A") == 1
end

@testset "Dynamic identities and model filters" begin
    ps = _query_fixture()
    dyn = ps.dynamic
    @test find_devices(ps) == collect(eachindex(dyn.devices))
    machines = find_devices(ps; model="GENROU")
    @test length(machines) == 2
    @test find_devices(ps; model=:GeNrOu) == machines
    @test find_devices(ps; model=:gensal) == machines
    @test find_devices(ps; model=GradPower.Genrou) == machines
    @test find_devices(ps; model=GradPower.AbstractGeneratorType) == machines
    @test find_devices(ps; model=GradPower.AbstractDeviceType) == find_devices(ps)
    @test find_devices(ps; model=:unregistered) == Int[]
    @test_throws ArgumentError find_devices(ps; model=Int)
    @test_throws ArgumentError find_devices(ps; online=1)
    @test_throws ArgumentError find_devices(ps; id=1)
    a = find_device(ps, :genrou, 101, " 'A ' ")
    b = find_device(ps, "GENSAL", 1, "B"; numbering=:internal)
    @test a != b
    @test find_devices(ps; bus=101, id="A") ==
          sort([a, find_device(ps, :tgov1, 101, "A")])
    @test find_devices(ps; bus=1, numbering=:internal) ==
          find_devices(ps; bus=101)
    dyn.devices[a].online = false
    @test find_devices(ps; model=:genrou, online=false) == [a]
    @test find_devices(ps; model=:genrou, online=true) == [b]
    @test find_device(ps, :genrou, 101, "A") == a
    @test_throws ArgumentError find_device(ps, :genrou, 101, "missing")

    # Bus filters use the dynamic map, independent of each dtype's bus convention.
    load = find_device(ps, :zipload, 7, "L")
    @test dyn.devices[load].dtype.bus == 2
    sg = only(find_devices(ps; model=:static_gen))
    aggregate = GradPower._result_device_id(ps, dyn.devices[sg].dtype)
    @test aggregate == "[\"C\",\"D\"]"
    @test find_device(ps, "STATICGENERATOR", 305, aggregate) == sg
    @test find_device(ps, GradPower.StaticGenerator, 3, aggregate;
                      numbering=:internal) == sg
    @test isempty(find_devices(ps; model=:static_gen, id="C"))
    @test_throws ArgumentError find_device(ps, :static_gen, 305, "C")
    members = copy(dyn.devices[sg].dtype.gen_idxs)
    resize!(dyn.devices[sg].dtype.gen_idxs, 1)
    @test find_device(ps, :static_gen, 305, "'C '") == sg
    append!(empty!(dyn.devices[sg].dtype.gen_idxs), members)

    dyn.devices[b].dtype.id = "'A'"
    @test length(find_devices(ps; model=:genrou, bus=101, id="A")) == 2
    @test_throws ArgumentError find_device(ps, :genrou, 101, "A")
    @test_throws ArgumentError state_index(ps, :genrou, 101, "A", :w)
end

@testset "Current state and parameter indices after cluster reorder" begin
    ps = _query_fixture()
    dyn = ps.dynamic
    channels = state_channels(ps)
    @test any(d -> d.alg_ptr <= 0, dyn.devices)
    @test any(d -> d.diff_ptr > dyn.diff_dim && d.dtype.diff_size > 0, dyn.devices)
    for channel in channels
        @test state_index(ps, channel.model, channel.bus, channel.device_id,
                          channel.name) == channel.index
        @test state_index(ps, uppercase(String(channel.model)), channel.bus,
                          channel.device_id, String(channel.name)) == channel.index
        @test get_index_info(channel.index, ps) == channel
    end
    @test state_index(ps, :bus, 2, "", :vi; numbering=:internal) ==
          dyn.diff_dim + dyn.alg_dim + 4
    @test state_index(ps, GradPower.Genrou, 101, "'A '", "w") ==
          state_index(ps, :gensal, 101, "A", :w)
    @test_throws ArgumentError state_index(ps, :genrou, 101, "A", :W)
    @test_throws ArgumentError state_index(ps, :bus, 7, "L", :vr)
    @test_throws ArgumentError state_index(ps, :bus, 7, "", :v)
    @test_throws BoundsError get_index_info(0, ps)
    @test_throws BoundsError get_index_info(-1, ps)
    @test_throws BoundsError get_index_info(length(channels) + 1, ps)
    for (i, dev) in enumerate(dyn.devices), (offset, name) in enumerate(get_param_names(dev.dtype))
        bus = ps.buses[dyn.map.bus[i]].i
        id = GradPower._result_device_id(ps, dev.dtype)
        @test parameter_index(ps, i, name) == dev.par_ptr + offset - 1
        @test parameter_index(ps, i, Symbol(name)) == dev.par_ptr + offset - 1
        @test parameter_index(ps, get_device_name(dev.dtype), bus, id, name) ==
              dev.par_ptr + offset - 1
    end
    governor = find_device(ps, :tgov1, 101, "A")
    @test_throws ArgumentError parameter_index(ps, governor, :r)
    @test_throws ArgumentError parameter_index(ps, governor, :missing)
    @test_throws ArgumentError parameter_index(ps, governor, 1)
    @test_throws BoundsError parameter_index(ps, 0, :R)
    @test_throws BoundsError parameter_index(ps, length(dyn.devices) + 1, :R)
    dyn.devices[governor].par_ptr += 10
    @test parameter_index(ps, :tgov1, 101, "A", :R) == dyn.devices[governor].par_ptr

    rebuilt = _query_fixture(reverse_machines=true)
    @test state_index(ps, :genrou, 101, "A", :w) !=
          state_index(rebuilt, :genrou, 101, "A", :w)
    for channel in state_channels(rebuilt)
        @test get_index_info(state_index(rebuilt, channel.model, channel.bus,
                                        channel.device_id, channel.name), rebuilt) == channel
    end
end

@testset "Plain summaries and IO display" begin
    ps = _query_fixture()
    @test getproperty.(system_summary(ps), :bus) == [101, 7, 305]
    for section in (:buses, :generators, :loads, :devices)
        rows = system_summary(ps; section)
        @test rows isa Vector{NamedTuple}
        @test all(r -> r.bus in (101, 7, 305), rows)
        @test getproperty.(rows, :index) == collect(eachindex(rows))
    end
    @test system_summary(ps; section=:generators)[1].id == "A"
    @test only(system_summary(ps; section=:loads)).bus == 7
    @test_throws ArgumentError system_summary(ps; section=:branches)
    io = IOBuffer()
    @test display_parameters(io, ps) === nothing
    output = String(take!(io))
    @test occursin("Device 1: $(get_device_name(ps.dynamic.devices[1].dtype))", output)
    @test occursin("[$(parameter_index(ps, :tgov1, 101, "A", :R))] R", output)
    last_channel = last(state_channels(ps))
    @test get_index_info(io, last_channel.index, ps) == last_channel
    output = String(take!(io))
    @test occursin("bus=305", output)
    @test occursin("vi", output)
    @test get_bus_info(io, 7, ps) === nothing
    output = String(take!(io))
    @test occursin("Connected Loads:", output)
    @test occursin("bus = 7", output)
    @test occursin("Connected Devices:", output)
    @test occursin("fr = 101, to = 7", output)
    @test occursin("Connected Shunts:", output)
    @test_throws ArgumentError get_bus_info(io, 2, ps)
    show(io, ps)
    output = String(take!(io))
    @test occursin("101\tA\t", output)
    @test occursin("7\tL\t", output)
    @test occursin("101\t7\t", output)
    @test occursin("7\t'S '\t", output)

    # Redirect stdout as well to detect accidental writes outside the supplied IO.
    mktemp() do _, captured
        redirect_stdout(captured) do
            display_parameters(io, ps)
            get_bus_info(io, 101, ps)
            get_index_info(io, 1, ps)
            get_index_info(1, ps)
        end
        flush(captured)
        seekstart(captured)
        @test isempty(read(captured, String))
        redirect_stdout(captured) do
            display_parameters(ps)
            get_bus_info(101, ps)
        end
        flush(captured)
        seekstart(captured)
        wrappers = read(captured, String)
        @test occursin("Displaying parameters", wrappers)
        @test occursin("Connected Generators", wrappers)
    end
    static = _query_fixture(dynamics=false)
    @test isempty(find_devices(static))
    @test isempty(system_summary(static; section=:devices))
    @test_throws ArgumentError find_device(static, :genrou, 101, "A")
    @test_throws ArgumentError state_index(static, :bus, 101, "", :vr)
    @test_throws ArgumentError parameter_index(static, 1, :R)
    empty_io = IOBuffer()
    display_parameters(empty_io, static)
    @test occursin("No dynamic data found.", String(take!(empty_io)))
    get_bus_info(empty_io, 7, static)
    @test occursin("Connected Loads:", String(take!(empty_io)))
end
