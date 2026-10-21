using Test
using GradPower

function _result_setup(raw="ieee9_v33.raw", dyr="ieee9bus_gov.dyr")
    ex = joinpath(@__DIR__, "..", "examples")
    ps = from_psse(joinpath(ex, raw), joinpath(ex, dyr))
    build_network!(ps)
    runpf!(ps)
    dp = DynamicProblem(ps)
    initialize_dynamics!(dp, ps)
    return ps, dp
end

@testset "SimulationResult" begin
    ps, dp = _result_setup()
    z0 = copy(dp.zvec)
    tuple_output = integrate!(dp, ps, 0.05)
    @test tuple_output isa Tuple
    dp.zvec .= z0
    r = integrate!(dp, ps, 0.05; result=true, scenario_ids=["base"])
    @test r isa SimulationResult
    @test r.times == tuple_output[1]
    @test r.trajectories == tuple_output[2]
    @test Tuple(r) == tuple_output
    @test r.status == [:success]
    @test r.scenario_ids == ["base"]
    @test r.metadata.final_time == 0.05
    @test r.storage === nothing
    @test isempty(r.events)
    @test length(r.channels) == length(z0)
    @test getfield.(r.channels, :index) == eachindex(z0)
    @test count(c -> c.kind == :differential, r.channels) == ps.dynamic.diff_dim
    for c in r.channels
        @test state(r, c.model, c.bus, c.device_id, c.name) ==
              r.trajectories[c.index, :]
    end
    c = first(filter(c -> c.model == :genrou && c.name == :w, r.channels))
    w = speed(r, c.bus, " '$(c.device_id)' ")
    @test w isa SubArray
    @test w == state(r, :genrou, 1, :w)
    @test w == speed(r, 1)
    @test_throws ArgumentError state(r, :genrou, 999, :w)
    @test_throws ArgumentError state(r, :genrou, 1, :unknown)
    @test_throws ArgumentError speed(r, 1; scenario="missing")
    bus = ps.buses[1].i
    vr = bus_voltage(r, bus; component=:real)
    vi = bus_voltage(r, bus; component=:imag)
    @test vr isa SubArray
    @test bus_voltage(r, bus) == complex.(vr, vi)
    @test bus_voltage(r, bus; component=:magnitude) == hypot.(vr, vi)
    @test bus_voltage(r, bus; component=:angle) == atan.(vi, vr)
    @test_throws ArgumentError bus_voltage(r, bus; component=:missing)

    @testset "selected rows and schema snapshots" begin
        rows = [c.index, c.index, lastindex(z0)]
        dp.zvec .= z0
        sub = integrate!(dp, ps, 0.05; result=true, save_idxs=rows,
                         save_every=2, save_eltype=Float32)
        @test sub.trajectories == Float32.(r.trajectories[rows, 1:2:end])
        @test getfield.(sub.channels, :index) == rows
        @test speed(sub, 1) == sub.trajectories[1, :]
        @test_throws ArgumentError bus_voltage(sub, bus)
        snapshot = copy(sub.channels)
        ps.dynamic.devices[1].diff_ptr += 100
        @test sub.channels == snapshot
        @test speed(sub, 1) == sub.trajectories[1, :]
        ps.dynamic.devices[1].diff_ptr -= 100
        dp.zvec .= z0
        empty_result = integrate!(dp, ps, 0.0; result=true, save_idxs=Int[])
        @test isempty(empty_result.channels)
        @test size(empty_result.trajectories) == (0, 1)
    end

    @testset "event metadata and early termination" begin
        dt = 1 / 120
        add_event!(ps, ContingencyEvent(4, 0.1, 2dt, 4dt))
        dp.zvec .= z0
        full = integrate!(dp, ps, 0.05; result=true)
        @test getproperty.(full.events, :action) == [:fault_on, :fault_off]
        @test full.events[1].details.bus == ps.buses[4].i
        dp.zvec .= z0
        early = integrate!(dp, ps, 0.05; result=true, save_every=4,
                           stop_condition=(k, t, z) -> k == 4)
        @test early.status == [:terminated]
        @test early.metadata.final_time == 3dt
        @test early.times == [0.0]
        @test getproperty.(early.events, :action) == [:fault_on]
        dp.zvec .= z0
        before = integrate!(dp, ps, 0.05; result=true,
                            stop_condition=(k, t, z) -> k == 3)
        @test isempty(before.events) # stops before this step's fault activation
        empty!(ps.dynamic.events)
        # The explicit-fault GPU backend ignores the registered schedule.
        # Its context must do so too, even for an unbounded global fault.
        add_event!(ps, ContingencyEvent(4, 0.1, 0.1, Inf))
        context = GradPower._result_context(ps, :, 2, nothing, dt, (;); system_events=false)
        @test isempty(context.events)
        empty!(ps.dynamic.events)
    end

    @testset "batched scenarios" begin
        dp.zvec .= z0
        bl = GradPower.BatchedLayout(dp, ps, 2)
        batch = GradPower.integrate_batched!(bl, ps, 0.05; result=true,
                                            scenario_ids=[:a, :b], threaded=false)
        @test batch.scenario_ids == [:a, :b]
        @test batch.status == [:success, :success]
        @test batch.trajectories[1] == r.trajectories
        @test speed(batch, 1; scenario=:b) == speed(r, 1)
        @test_throws ArgumentError GradPower.integrate_batched!(bl, ps, 0.0;
            result=true, scenario_ids=[:a, :a])
    end

    @testset "constructor validation" begin
        @test_throws DimensionMismatch SimulationResult(r.times, zeros(1, 1), r.channels)
        @test_throws DimensionMismatch SimulationResult(r.times, r.trajectories, r.channels;
                                                        scenario_ids=[1, 2])
        @test_throws DimensionMismatch SimulationResult(r.times, r.trajectories, r.channels;
                                                        status=Symbol[])
        restored = SimulationResult(r.times, r.trajectories, r.channels;
                                    scenario_ids=r.scenario_ids, storage="caller-owned")
        @test speed(restored, 1) == speed(r, 1)
        @test restored.storage == "caller-owned"
        ambiguous = [c, StateChannel(c.index + 1, c.model, c.bus, c.device_id,
                                     c.name, c.kind, c.online)]
        bad = SimulationResult([0.0], zeros(2, 1), ambiguous)
        @test_throws ArgumentError state(bad, c.model, c.bus, c.device_id, c.name)
    end
end

@testset "Results own only this run's limit events" begin
    for method in (:active_set, :complementarity)
        ps, dp = _result_setup("2bus.raw", "2bus_TGOV1.dyr")
        for dev in ps.dynamic.devices
            dev.dtype isa GradPower.ZIPLoad && (dev.dtype.α = 0.5)
        end
        initialize_dynamics!(dp, ps)
        table = ps.dynamic.layout.tgov1
        pp = Int(table.par_ptr[1])
        st = Int(table.diff_ptr[1]) + 1
        dp.pvec[pp + 2] = dp.zvec[st] + 0.01
        dp.pvec[pp + 3] = dp.zvec[st] - 0.01
        dp.pvec[pp + 7] += 0.05
        z0 = copy(dp.zvec)
        events = LimitEvent[]
        first_run = integrate!(dp, ps, 0.05; dt=0.005, result=true,
                               limit_method=method, limit_events=events)
        @test !isempty(events)
        @test first_run.metadata.limit_events == events
        old_events = copy(events)
        dp.zvec .= z0
        zero_run = integrate!(dp, ps, 0.0; dt=0.005, result=true,
                              limit_method=method, limit_events=events)
        @test isempty(zero_run.metadata.limit_events)
        @test events == old_events
        dp.zvec .= z0
        again = integrate!(dp, ps, 0.05; dt=0.005, result=true,
                           limit_method=method, limit_events=events)
        @test again.metadata.limit_events == old_events
        @test first_run.metadata.limit_events == old_events
        @test length(events) == 2length(old_events)
    end
end

@testset "State schema for every native model and offline devices" begin
    for dyr in ("2bus.dyr", "2bus_IEESGO.dyr", "2bus_TGOV1.dyr",
                "2bus_SEXS.dyr", "2bus_ESDC1A.dyr", "2bus_IEEEST.dyr")
        ps, dp = _result_setup("2bus.raw", dyr)
        channels = state_channels(ps)
        @test length(channels) == length(dp.zvec)
        @test allunique(getfield.(channels, :index))
        for (i, dev) in enumerate(ps.dynamic.devices)
            model = Symbol(lowercase(get_device_name(dev.dtype)))
            bus = ps.buses[ps.dynamic.map.bus[i]].i
            cs = filter(c -> c.model == model && c.bus == bus, channels)
            @test length(cs) == dev.dtype.diff_size + dev.dtype.alg_size
            @test Set(string(c.name) for c in cs if c.kind == :differential) ==
                  Set(get_diff_names(dev.dtype))
            @test Set(string(c.name) for c in cs if c.kind == :algebraic) ==
                  Set(get_alg_names(dev.dtype))
        end
        idx = findfirst(d -> d.dtype isa GradPower.Genrou, ps.dynamic.devices)
        add_disconnect_event!(ps, DisconnectDeviceEvent(idx, 1.0))
        GradPower._set_device_offline!(ps.dynamic, idx)
        offline = state_channels(ps)
        @test any(c -> c.model == :genrou && !c.online, offline)
        @test all(c -> c.online, channels)
        frozen = integrate!(dp, ps, 0.0; result=true)
        @test speed(frozen, 1) == [dp.zvec[ps.dynamic.devices[idx].diff_ptr + 4]]
    end
end

@testset "Co-located devices use stable identities" begin
    ps, _ = _result_setup("2bus.raw", "2bus.dyr")
    bus = 101
    ps.buses[1].i = bus
    empty!(ps.busmap)
    for (i, b) in enumerate(ps.buses)
        ps.busmap[b.i] = i
    end
    # Include an aggregated static generator with two member IDs.
    gen = deepcopy(first(ps.gens))
    gen.id = "'2 '"
    push!(ps.gens, gen)
    dyn = PowerSystemDynamics()
    add_device!(dyn, GradPower.GenericGenerator(bus, "'B '"))
    add_device!(dyn, GradPower.GenericGenerator(bus, "'A '"))
    set_dynamics!(ps, dyn)
    channels = state_channels(ps)
    values = reshape(Float64.(eachindex(channels)), :, 1)
    r = SimulationResult([0.0], values, channels)
    @test speed(r, bus, "A") != speed(r, bus, "B")
    @test speed(r, 1) == speed(r, bus, "A")
    @test speed(r, 2) == speed(r, bus, "B")
    @test all(c -> c.bus in getfield.(ps.buses, :i), channels)
    aggregate = filter(c -> c.model == :staticgenerator, channels)
    @test any(c -> startswith(c.device_id, "["), aggregate)
    for c in aggregate
        @test state(r, c.model, c.bus, c.device_id, c.name) == values[c.index, :]
    end

    # Rebuild with a different physical row order but the same identities.
    rebuilt = PowerSystemDynamics()
    add_device!(rebuilt, GradPower.GenericGenerator(bus, "'A '"))
    add_device!(rebuilt, GradPower.GenericGenerator(bus, "'B '"))
    set_dynamics!(ps, rebuilt)
    reordered = state_channels(ps)
    key(c) = (c.model, c.bus, c.device_id, c.name)
    by_identity = Dict(key(c) => values[c.index, 1] for c in channels)
    new_values = reshape([by_identity[key(c)] for c in reordered], :, 1)
    r2 = SimulationResult([0.0], new_values, reordered)
    @test speed(r2, bus, "A") == speed(r, bus, "A")
    @test speed(r2, bus, "B") == speed(r, bus, "B")
    @test findfirst(c -> key(c) == (:genrou, bus, "A", :w), channels) !=
          findfirst(c -> key(c) == (:genrou, bus, "A", :w), reordered)
    @test bus_voltage(r2, bus) == bus_voltage(r, bus)
end
