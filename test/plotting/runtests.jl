using Test
using GradPower
using CairoMakie

CairoMakie.activate!()
include(joinpath(@__DIR__, "..", "test_plotting.jl"))

_render_axes(fig) = filter(x -> x isa CairoMakie.Axis, fig.content)

function _test_render_files(fig, directory, stem)
    @test fig isa CairoMakie.Figure
    pixels = CairoMakie.colorbuffer(fig)
    @test !isempty(pixels)
    @test length(unique(pixels)) > 16
    for extension in ("png", "svg")
        path = joinpath(directory, "$stem.$extension")
        CairoMakie.save(path, fig)
        @test isfile(path)
        @test filesize(path) > 256
        if extension == "png"
            @test read(path)[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        else
            @test occursin("<svg", read(path, String))
        end
    end
end

@testset "Headless result figures" begin
    r = _plot_test_result(factors=(1.0, 2.0, 3.0), ids=[41, 7, 99])
    original = deepcopy(r.trajectories)
    mktempdir() do directory
        selectors = [(:genrou, 101, "A", :w), (:bus, 7, "", :vr)]
        states = plot_states(r, selectors; scenarios=[99, 41],
                             figure_kwargs=(size=(640, 480),),
                             axis_kwargs=(xlabel="Elapsed (s)",))
        axes = _render_axes(states)
        @test length(axes) == 2
        @test all(ax -> ax.xlabel[] == "Elapsed (s)", axes)
        @test all(ax -> length(ax.scene.plots) == 2, axes)
        _test_render_files(states, directory, "states")
        channel_states = plot_states(r, [r.channels[2]]; scenarios=[41])
        @test length(_render_axes(channel_states)) == 1
        @test length(only(_render_axes(channel_states)).scene.plots) == 1
        mixed = _plot_test_result(factors=(1.0, 2.0), ids=Any[1, "1"])
        mixed_states = plot_states(mixed, [mixed.channels[2]])
        mixed_labels = [p.label[] for p in only(_render_axes(mixed_states)).scene.plots
                        if p isa CairoMakie.Lines]
        @test mixed_labels == ["Scenario $(repr(id))" for id in mixed.scenario_ids]
        @test length(unique(mixed_labels)) == 2

        voltage = plot_voltage(r; buses=[101, 7], scenarios=[41])
        @test length(_render_axes(voltage)) == 1
        @test length(only(_render_axes(voltage)).scene.plots) == 2
        _test_render_files(voltage, directory, "voltage")
        for component in (:real, :imag, :angle)
            fig = plot_voltage(r; buses=[101], scenarios=[99], component)
            @test length(only(_render_axes(fig)).scene.plots) == 1
            component === :angle && @test occursin("rad", lowercase(only(_render_axes(fig)).ylabel[]))
        end
        envelope = plot_voltage(r; buses=[101], scenarios=[99, 41], envelope=true)
        @test length(only(_render_axes(envelope)).scene.plots) == 2
        _test_render_files(envelope, directory, "voltage-envelope")

        speed = plot_speed(r; generators=[(101, "B")], scenarios=[99, 41],
                           unit=:hz, nominal_frequency=50.0)
        speed_axis = only(_render_axes(speed))
        @test occursin("hz", lowercase(speed_axis.ylabel[]))
        @test occursin("deviation", lowercase(speed_axis.ylabel[]))
        @test length(speed_axis.scene.plots) == 2
        _test_render_files(speed, directory, "speed-hz")
        speed_envelope = plot_speed(r; generators=[(101, "A")], scenarios=[41, 99],
                                    envelope=true)
        @test length(only(_render_axes(speed_envelope)).scene.plots) == 2
        _test_render_files(speed_envelope, directory, "speed-envelope")

        source = _plot_test_result()
        singleton = SimulationResult([0.25], source.trajectories[:, 2:2], source.channels)
        single_figures = [
            ("state", plot_states(singleton, [singleton.channels[2]])),
            ("voltage", plot_voltage(singleton; buses=[101])),
            ("speed", plot_speed(singleton; generators=[(101, "A")])),
        ]
        for (name, fig) in single_figures
            @test any(p -> p isa CairoMakie.Scatter, only(_render_axes(fig)).scene.plots)
            _test_render_files(fig, directory, "singleton-$name")
        end
    end
    @test r.trajectories == original
end

@testset "Headless network and animation" begin
    ps = _plot_test_system()
    r = _plot_test_result(factors=(1.0, 2.0), ids=[41, 99])
    positions = Dict(101 => (0.0, 1.0), 7 => (-1.0, 0.0), 305 => (1.0, 0.0))
    mktempdir() do directory
        static = plot_network(ps; positions, figure_kwargs=(size=(640, 400),))
        @test length(_render_axes(static)) == 1
        _test_render_files(static, directory, "network-static")
        network = plot_network(ps, r; scenario=99, frame=2, positions,
                               labels=false, quantity=:angle,
                               colorrange=(-pi, pi), colormap=:plasma)
        @test length(_render_axes(network)) == 1
        @test occursin("0.25", string(only(_render_axes(network)).title[]))
        nodes = only(filter(p -> p isa CairoMakie.Scatter,
                            only(_render_axes(network)).scene.plots))
        @test isapprox(collect(nodes.colorrange[]), [-pi, pi]; rtol=1e-6)
        expected = GradPower._plot_network_data(ps, r; scenario=99, frame=2,
                                                positions, quantity=:angle)
        @test isapprox(nodes.color[], expected.values; rtol=1e-6)
        _test_render_files(network, directory, "network-angle")
        @test_throws ArgumentError plot_network(ps, r; colorrange=(2.0, 1.0))
        @test_throws ArgumentError plot_network(ps, r; colorrange=(NaN, 1.0))
        for fps in (0.1, 0.5, 1.5, 20.0, true)
            @test_throws ArgumentError animate_network(
                joinpath(directory, "invalid-framerate.gif"), ps, r; framerate=fps)
        end
        @test !isfile(joinpath(directory, "invalid-framerate.gif"))
        path = joinpath(directory, "network.gif")
        @test animate_network(path, ps, r; scenario=99, frames=[1, 3],
                               framerate=2, positions, colorrange=(0.0, 10.0),
                               figure_kwargs=(size=(320, 240),)) == path
        @test isfile(path)
        @test filesize(path) > 256
        @test String(read(path)[1:6]) in ("GIF87a", "GIF89a")
        auto_path = joinpath(directory, "network-auto.gif")
        @test animate_network(auto_path, ps, r; frames=[1, 2, 3], framerate=3,
                               labels=false, figure_kwargs=(size=(320, 240),)) == auto_path
        @test isfile(auto_path)
        @test filesize(auto_path) > 256
    end
end

include("cpf.jl")
include("load_order.jl")
