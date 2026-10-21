using Test

@testset "Fresh-process plotting load order" begin
    project = dirname(something(Base.active_project(), joinpath(@__DIR__, "Project.toml")))
    for order in (:core_only, :core_first, :backend_first)
        script = if order === :core_only
            """
            using Test, GradPower
            @test Base.get_extension(GradPower, :GradPowerMakieExt) === nothing
            r = SimulationResult([0.0], ones(1, 1),
                [StateChannel(1, :genrou, 101, "A", :w, :differential, true)])
            ps = PowerSystem(100.0, [Bus(101, "A", 3, 230.0, 1.0, 0.0)],
                Gen[], Load[], Branch[], Shunt[], Dict(101 => 1))
            calls = [
                () -> plot_states(r, [(:genrou, 101, "A", :w)]),
                () -> plot_voltage(r),
                () -> plot_speed(r),
                () -> plot_network(ps),
                () -> animate_network("unused.gif", ps, r),
            ]
            for call in calls
                err = try
                    call()
                    nothing
                catch error
                    error
                end
                @test err isa ArgumentError
                if err isa ArgumentError
                    @test occursin("CairoMakie", sprint(showerror, err))
                end
            end
            @test !any(id -> id.name == "Makie" || id.name == "CairoMakie",
                       keys(Base.loaded_modules))
            """
        else
            imports = order === :core_first ?
                "using GradPower; using CairoMakie" : "using CairoMakie; using GradPower"
            """
            using Test
            $imports
            CairoMakie.activate!()
            @test Base.get_extension(GradPower, :GradPowerMakieExt) !== nothing
            r = SimulationResult([0.0, 1.0], [0.0 0.01],
                [StateChannel(1, :genrou, 101, "A", :w, :differential, true)])
            fig = plot_speed(r)
            @test fig isa CairoMakie.Figure
            @test length(unique(CairoMakie.colorbuffer(fig))) > 16
            """
        end
        command = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $script`
        @test success(command)
    end
end
