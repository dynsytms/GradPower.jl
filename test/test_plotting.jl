using Test
using GradPower

function _plot_test_result(; factors=(1.0,), ids=[41], times=[0.0, 0.25, 1.0])
    channels = [
        StateChannel(31, :bus, 101, "", :vr, :algebraic, true),
        StateChannel(5, :genrou, 101, "A", :w, :differential, true),
        StateChannel(42, :bus, 7, "", :vi, :algebraic, true),
        StateChannel(32, :bus, 101, "", :vi, :algebraic, true),
        StateChannel(15, :genrou, 101, "B", :w, :differential, true),
        StateChannel(41, :bus, 7, "", :vr, :algebraic, true),
        StateChannel(25, :genrou, 305, "C", :w, :differential, true),
        StateChannel(51, :bus, 305, "", :vr, :algebraic, true),
        StateChannel(52, :bus, 305, "", :vi, :algebraic, true),
        StateChannel(5, :genrou, 101, "A", :w, :differential, true),
    ]
    values = [3.0 0.0 -3.0;
              0.0 0.01 -0.02;
              0.0 2.0 0.0;
              4.0 2.0 4.0;
              0.02 -0.01 0.0;
              1.0 0.0 -1.0;
              -0.01 0.0 0.01;
              1.0 2.0 3.0;
              0.0 0.0 0.0;
              0.0 0.01 -0.02]
    matrices = [factor .* values for factor in factors]
    trajectories = length(matrices) == 1 ? only(matrices) : matrices
    return SimulationResult(times, trajectories, channels; scenario_ids=ids)
end

function _plot_test_system(; reverse_buses=false)
    buses = [Bus(101, "NORTH", 3, 230.0, 1.1, 0.2),
             Bus(7, "SOUTH", 1, 230.0, 0.9, -0.1),
             Bus(305, "EAST", 1, 230.0, 1.0, 0.0)]
    reverse_buses && reverse!(buses)
    busmap = Dict(b.i => i for (i, b) in enumerate(buses))
    branches = [Branch(busmap[101], busmap[7], "1", 0.01, 0.1, 0.0, 0.0, 0.0),
                Branch(busmap[7], busmap[305], "1", 0.01, 0.1, 0.0, 0.0, 0.0)]
    return PowerSystem(100.0, buses, Gen[], Load[], branches, Shunt[], busmap)
end

mutable struct _PlotReadMatrix <: AbstractMatrix{Float64}
    data::Matrix{Float64}
    reads::Vector{Tuple{Int,Int}}
end
Base.size(a::_PlotReadMatrix) = size(a.data)
Base.IndexStyle(::Type{_PlotReadMatrix}) = IndexCartesian()
function Base.getindex(a::_PlotReadMatrix, i::Int, j::Int)
    push!(a.reads, (i, j))
    return a.data[i, j]
end

mutable struct _PlotReadVector <: AbstractVector{Float64}
    data::Vector{Float64}
    reads::Int
end
Base.size(a::_PlotReadVector) = size(a.data)
Base.IndexStyle(::Type{_PlotReadVector}) = IndexLinear()
function Base.getindex(a::_PlotReadVector, i::Int)
    a.reads += 1
    return a.data[i]
end

@testset "Plot helper times and raw states without Makie" begin
    r = _plot_test_result()
    @test GradPower._plot_times(r) == [0.0, 0.25, 1.0]
    for times in ([0.0, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, NaN, 1.0],
                  [0.0, Inf, 1.0])
        invalid = SimulationResult(times, copy(r.trajectories), r.channels;
                                   scenario_ids=[41])
        @test_throws ArgumentError GradPower._plot_times(invalid)
    end
    empty_result = SimulationResult(Float64[], zeros(length(r.channels), 0), r.channels)
    @test_throws ArgumentError GradPower._plot_times(empty_result)
    selectors = [(:genrou, 101, "'A '", :w), (:bus, 7, "", :vi)]
    data = GradPower._plot_states_data(r, selectors)
    @test length(data) == 2
    @test data[1].series[1].scenario == 41
    @test data[1].series[1].values == [0.0, 0.01, -0.02]
    @test data[2].series[1].values == [0.0, 2.0, 0.0]
    @test all(d -> !isempty(d.label) && !isempty(d.ylabel), data)
    by_channel = GradPower._plot_states_data(r, [r.channels[2], r.channels[3]])
    @test [d.series[1].values for d in by_channel] ==
          [d.series[1].values for d in data]
    @test_throws ArgumentError GradPower._plot_states_data(r, Tuple[])
    @test_throws ArgumentError GradPower._plot_states_data(r, [(:genrou, 101, "Z", :w)])
    @test_throws ArgumentError GradPower._plot_states_data(r, [(:genrou, 101, "A", :W)])
    @test_throws ArgumentError GradPower._plot_states_data(r, selectors; scenarios=[])
    @test_throws ArgumentError GradPower._plot_states_data(r, selectors; scenarios=[41, 41])
    @test_throws ArgumentError GradPower._plot_states_data(r, selectors; scenarios=[1])
    @test_throws ArgumentError GradPower._plot_states_data(r, selectors; scenarios=41)
    batched = _plot_test_result(factors=(1.0, 2.0, 3.0), ids=[41, 7, 99])
    chosen = GradPower._plot_states_data(batched, selectors; scenarios=[99, 41])
    @test getproperty.(chosen[1].series, :scenario) == [99, 41]
    @test chosen[1].series[1].values == [0.0, 0.03, -0.06]
    @test length(only(GradPower._plot_states_data(batched, selectors[1:1])).series) == 3
    named = _plot_test_result(factors=(1.0, 2.0), ids=["base", "storm"])
    named_data = GradPower._plot_states_data(named, selectors[1:1]; scenarios=["storm"])
    @test only(only(named_data).series).scenario == "storm"
    @test only(only(named_data).series).values == [0.0, 0.02, -0.04]
    mixed = _plot_test_result(factors=(1.0, 2.0), ids=Any[1, "1"])
    mixed_series = only(GradPower._plot_states_data(mixed, selectors[1:1])).series
    @test length(mixed_series) == 2
    @test mixed_series[1].scenario === 1
    @test mixed_series[2].scenario == "1"
    @test mixed_series[2].values == [0.0, 0.02, -0.04]

    # Duplicate saved rows share an identity; different raw indices do not.
    conflicting = copy(r.channels)
    conflicting[end] = StateChannel(999, :genrou, 101, "A", :w, :differential, true)
    ambiguous = SimulationResult(r.times, copy(r.trajectories), conflicting)
    @test_throws ArgumentError GradPower._plot_states_data(ambiguous, selectors[1:1])
    subset = SimulationResult(r.times, r.trajectories[[5, 2], :], r.channels[[5, 2]])
    @test only(GradPower._plot_states_data(subset, selectors[1:1])).series[1].values ==
          [0.0, 0.01, -0.02]
    @test_throws ArgumentError GradPower._plot_states_data(subset, selectors[2:2])
end

@testset "Voltage components and scenario envelopes" begin
    r = _plot_test_result(factors=(1.0, 2.0, 3.0), ids=[41, 7, 99])
    for (component, expected) in ((:magnitude, [5.0, 2.0, 5.0]),
                                  (:real, [3.0, 0.0, -3.0]),
                                  (:imag, [4.0, 2.0, 4.0]),
                                  (:angle, atan.([4.0, 2.0, 4.0], [3.0, 0.0, -3.0])))
        data = GradPower._plot_voltage_data(r; buses=[101], scenarios=[41], component)
        @test length(data.curves) == 1
        @test only(data.curves).values == expected
        @test only(data.curves).lower === nothing
        @test only(data.curves).upper === nothing
        @test !isempty(only(data.curves).label)
    end
    @test occursin("rad", lowercase(GradPower._plot_voltage_data(r; component=:angle).ylabel))
    ordinary = GradPower._plot_voltage_data(r; buses=[101, 7], scenarios=[99, 41])
    @test length(ordinary.curves) == 4
    envelope = GradPower._plot_voltage_data(r; buses=[101], scenarios=[99, 41],
                                           envelope=true)
    curve = only(envelope.curves)
    @test curve.values == [10.0, 4.0, 10.0]
    @test curve.lower == [5.0, 2.0, 5.0]
    @test curve.upper == [15.0, 6.0, 15.0]
    singleton = only(GradPower._plot_voltage_data(r; buses=[7], scenarios=[7],
                                                 envelope=true).curves)
    @test singleton.values == singleton.lower == singleton.upper == [2.0, 4.0, 2.0]
    @test_throws ArgumentError GradPower._plot_voltage_data(r; buses=[])
    @test_throws ArgumentError GradPower._plot_voltage_data(r; buses=[999])
    @test_throws ArgumentError GradPower._plot_voltage_data(r; buses=[101, 101])
    @test_throws ArgumentError GradPower._plot_voltage_data(r; scenarios=[99, 99])
    @test_throws ArgumentError GradPower._plot_voltage_data(r; scenarios=[])
    @test_throws ArgumentError GradPower._plot_voltage_data(r; component=:complex)
    subset = SimulationResult(r.times, r.trajectories[1][[1], :], r.channels[[1]])
    @test only(GradPower._plot_voltage_data(subset; buses=[101], component=:real).curves).values ==
          [3.0, 0.0, -3.0]
    @test_throws ArgumentError GradPower._plot_voltage_data(subset; buses=[101])
end

@testset "Raw speed deviation and frequency-deviation units" begin
    r = _plot_test_result(factors=(1.0, 2.0, 3.0), ids=[41, 7, 99])
    data = GradPower._plot_speed_data(r; generators=[(101, "'A '")], scenarios=[41])
    @test only(data.curves).values == [0.0, 0.01, -0.02]
    @test only(data.curves).lower === nothing
    @test only(data.curves).upper === nothing
    hz = GradPower._plot_speed_data(r; generators=[(101, "A")], scenarios=[41],
                                    unit=:hz, nominal_frequency=50.0)
    @test only(hz.curves).values == [0.0, 0.5, -1.0]
    @test occursin("hz", lowercase(hz.ylabel))
    @test occursin("deviation", lowercase(hz.ylabel))
    for model in (:gensal, "GENROU", :GeNrOu)
        alias = GradPower._plot_speed_data(r; generators=[(101, "A")],
                                           scenarios=[41], model)
        @test only(alias.curves).values == only(data.curves).values
    end
    @test length(GradPower._plot_speed_data(r; scenarios=[41]).curves) == 3
    env = only(GradPower._plot_speed_data(r; generators=[(101, "A")],
                                         scenarios=[99, 41], envelope=true,
                                         unit=:hz, nominal_frequency=50.0).curves)
    @test env.values == [0.0, 1.0, -2.0]
    @test env.lower == [0.0, 0.5, -3.0]
    @test env.upper == [0.0, 1.5, -1.0]
    @test_throws ArgumentError GradPower._plot_speed_data(r; generators=[])
    @test_throws ArgumentError GradPower._plot_speed_data(r; generators=[(101, "missing")])
    @test_throws ArgumentError GradPower._plot_speed_data(r; generators=[(101, "A"), (101, "'A '")])
    @test_throws ArgumentError GradPower._plot_speed_data(r; scenarios=[1])
    @test_throws ArgumentError GradPower._plot_speed_data(r; scenarios=[])
    @test_throws ArgumentError GradPower._plot_speed_data(r; scenarios=[41, 41])
    @test_throws ArgumentError GradPower._plot_speed_data(r; unit=:rpm)
    for frequency in (0.0, -60.0, Inf, NaN)
        @test_throws ArgumentError GradPower._plot_speed_data(r; unit=:hz,
                                                             nominal_frequency=frequency)
    end
end

@testset "Network frames, external buses, and deterministic positions" begin
    ps = _plot_test_system()
    r = _plot_test_result(factors=(1.0, 2.0, 3.0), ids=[41, 7, 99])
    data = GradPower._plot_network_data(ps, r)
    @test Dict(zip(data.buses, data.values)) == Dict(101 => 5.0, 7 => 1.0, 305 => 3.0)
    @test data.time == 1.0
    @test all(p -> p isa Tuple{Float64,Float64}, data.positions)
    @test all(v -> v isa Float64, data.values)
    @test !isempty(data.label)
    @test Set(Set((data.buses[a], data.buses[b])) for (a, b) in data.edges) ==
          Set([Set((101, 7)), Set((7, 305))])
    reordered = GradPower._plot_network_data(_plot_test_system(reverse_buses=true), r)
    @test Dict(zip(data.buses, data.positions)) ==
          Dict(zip(reordered.buses, reordered.positions))
    @test all(p -> isapprox(hypot(p...), 1.0), data.positions)
    positions = Dict(101 => (2, 4), 7 => (-1, 0), 305 => (0, 2))
    selected = GradPower._plot_network_data(ps, r; scenario=99, frame=2, positions)
    @test selected.time == 0.25
    @test Dict(zip(selected.buses, selected.values)) == Dict(101 => 6.0, 7 => 6.0, 305 => 6.0)
    @test Dict(zip(selected.buses, selected.positions)) == positions
    angle = GradPower._plot_network_data(ps, r; frame=2, quantity=:angle)
    @test Dict(zip(angle.buses, angle.values)) == Dict(101 => pi / 2, 7 => pi / 2, 305 => 0.0)
    @test occursin("rad", lowercase(angle.label))
    static = GradPower._plot_network_data(ps)
    @test static.time === nothing
    @test Dict(zip(static.buses, static.values)) == Dict(b.i => b.v0m for b in ps.buses)
    static_angle = GradPower._plot_network_data(ps; quantity=:angle)
    @test Dict(zip(static_angle.buses, static_angle.values)) ==
          Dict(b.i => b.v0a for b in ps.buses)
    @test_throws ArgumentError GradPower._plot_network_data(ps, r; scenario=1)
    @test_throws ArgumentError GradPower._plot_network_data(ps, r; quantity=:real)
    for frame in (0, 4, 1.5, true, false)
        @test_throws ArgumentError GradPower._plot_network_data(ps, r; frame)
    end
    @test_throws ArgumentError GradPower._plot_network_data(ps, r; positions=Dict(101 => (0, 0)))
    invalid_positions = Dict{Int,Tuple{Float64,Float64}}(positions)
    invalid_positions[7] = (Inf, 0.0)
    @test_throws ArgumentError GradPower._plot_network_data(ps, r; positions=invalid_positions)
    subset = SimulationResult(r.times, r.trajectories[1][[1, 4], :], r.channels[[1, 4]])
    @test_throws ArgumentError GradPower._plot_network_data(ps, subset)

    counted = _PlotReadMatrix(copy(r.trajectories[1]), Tuple{Int,Int}[])
    lazy = SimulationResult(r.times, counted, r.channels)
    GradPower._plot_network_data(ps, lazy; frame=2)
    @test !isempty(counted.reads)
    @test all(read -> last(read) == 2, counted.reads)
    @test length(counted.reads) <= 2length(ps.buses)
    counted.data[1, 1] = NaN
    empty!(counted.reads)
    @test GradPower._plot_network_data(ps, lazy; frame=2).values == selected.values ./ 3
    @test all(read -> last(read) == 2, counted.reads)
end

@testset "Animation frame data does not rescan saved times" begin
    source = _plot_test_result()
    n = 128
    times = _PlotReadVector(collect(range(0.0; step=0.1, length=n)), 0)
    values = repeat(source.trajectories[:, 1:1], 1, n)
    r = SimulationResult(times, values, source.channels)
    ps = _plot_test_system()
    times.reads = 0
    frames = GradPower._plot_animation_frames(r, nothing, 20)
    validation_reads = times.reads
    @test frames == collect(1:n)
    @test n <= validation_reads <= 4n
    selected_times = [GradPower._plot_network_frame_data(ps, r; frame).time
                      for frame in frames[1:100]]
    @test selected_times == times.data[1:100]
    # One title-time read per frame is expected; a full validation scan is not.
    @test 100 <= times.reads - validation_reads <= 200
    @test times.reads <= 4n + 200
end

@testset "Animation validation, finite selected data, and no mutation" begin
    r = _plot_test_result()
    @test GradPower._plot_animation_frames(r, nothing, 20) == [1, 2, 3]
    @test GradPower._plot_animation_frames(r, [1, 3], 5) == [1, 3]
    @test GradPower._plot_animation_frames(r, nothing, 1) == [1, 2, 3]
    for frames in (Int[], [0], [4], [1, 1], [3, 1], [1.5], [nothing],
                   [true], [false], [1, nothing])
        @test_throws ArgumentError GradPower._plot_animation_frames(r, frames, 20)
    end
    for fps in (0, -1, Inf, NaN, true, false, 0.1, 0.5, 1.5, 20.0)
        @test_throws ArgumentError GradPower._plot_animation_frames(r, nothing, fps)
    end
    original_values = copy(r.trajectories)
    original_times = copy(r.times)
    original_channels = copy(r.channels)
    selectors = [(:genrou, 101, "A", :w)]
    GradPower._plot_states_data(r, selectors)
    GradPower._plot_voltage_data(r)
    GradPower._plot_speed_data(r)
    GradPower._plot_network_data(_plot_test_system(), r)
    @test r.trajectories == original_values
    @test r.times == original_times
    @test r.channels == original_channels
    for bad in (NaN, Inf)
        invalid = _plot_test_result()
        invalid.trajectories[2, 2] = bad
        @test_throws ArgumentError GradPower._plot_states_data(invalid, selectors)
        @test_throws ArgumentError GradPower._plot_speed_data(invalid; generators=[(101, "A")])
        invalid.trajectories[1, 2] = bad
        @test_throws ArgumentError GradPower._plot_voltage_data(invalid; buses=[101])
        @test_throws ArgumentError GradPower._plot_network_data(_plot_test_system(), invalid;
                                                               frame=2)
    end
    batched = _plot_test_result(factors=(1.0, 2.0), ids=[41, 99])
    batched.trajectories[2][2, 1] = NaN
    @test only(GradPower._plot_speed_data(batched; generators=[(101, "A")],
                                         scenarios=[41]).curves).values == [0.0, 0.01, -0.02]
    @test_throws ArgumentError GradPower._plot_speed_data(batched; generators=[(101, "A")],
                                                         scenarios=[99], envelope=true)
end

@testset "Singleton saved results" begin
    source = _plot_test_result()
    r = SimulationResult([0.25], source.trajectories[:, 2:2], source.channels;
                         scenario_ids=["single"])
    @test GradPower._plot_times(r) == [0.25]
    state_data = GradPower._plot_states_data(r, [r.channels[2]])
    @test only(only(state_data).series).values == [0.01]
    voltage = only(GradPower._plot_voltage_data(r; buses=[101], envelope=true).curves)
    @test voltage.values == voltage.lower == voltage.upper == [2.0]
    speed_data = only(GradPower._plot_speed_data(r; generators=[(101, "A")],
                                                unit=:hz, envelope=true).curves)
    @test speed_data.values == speed_data.lower == speed_data.upper == [0.6]
    network = GradPower._plot_network_data(_plot_test_system(), r)
    @test network.time == 0.25
    @test network.values == [2.0, 2.0, 2.0]
    @test GradPower._plot_animation_frames(r, nothing, 20) == [1]
    @test GradPower._plot_animation_frames(r, [1], 20) == [1]
    @test_throws ArgumentError GradPower._plot_animation_frames(r, [nothing], 20)
end

if Base.get_extension(GradPower, :GradPowerMakieExt) === nothing
    @testset "Core plotting fallbacks do not load Makie" begin
        r = _plot_test_result()
        ps = _plot_test_system()
        mktempdir() do directory
            path = joinpath(directory, "unused.gif")
            calls = [() -> plot_states(r, [r.channels[2]]),
                     () -> plot_voltage(r),
                     () -> plot_speed(r),
                     () -> plot_network(ps),
                     () -> animate_network(path, ps, r)]
            for call in calls
                error = try
                    call()
                    nothing
                catch caught
                    caught
                end
                @test error isa ArgumentError
                if error isa ArgumentError
                    @test occursin("using CairoMakie", sprint(showerror, error))
                end
            end
            @test !isfile(path)
        end
        @test !any(id -> id.name in ("Makie", "CairoMakie"), keys(Base.loaded_modules))
    end
end
