@testset "Powerflow 1" begin
    @testset "Case 9 tests" begin
        ps = create_case9()
        build_network!(ps)
        sol = runpf(ps, verbose=false)

        # note. the following vectors are extracted from a matpower
        # run on the same case. The values are slightly different.
        # Here we are only checking for large inaccuracies.

        vmat = [
            1.0000 + 0.0000im
            0.9858 + 0.1680im
            0.9965 + 0.0832im
            0.9861 - 0.0414im
            0.9731 - 0.0683im
            1.0028 + 0.0337im
            0.9856 + 0.0107im
            0.9940 + 0.0660im
            0.9549 - 0.0726im]
        vmag = sol.volt[1:2:end]
        vang = sol.volt[2:2:end]
        for i in 1:length(vmat)
            @test isapprox(abs(vmat[i]), vmag[i], rtol=1e-2)
            @test isapprox(angle(vmat[i]), vang[i], rtol=1e-2)
        end
    end
end

@testset "Slack generation includes local load" begin
    root = abspath(joinpath(@__DIR__, ".."))
    ps = GradPower.from_psse(joinpath(root, "examples", "IEEE39.raw"),
                             joinpath(root, "examples", "IEEE39_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)

    slack_bus = findfirst(bus -> bus.type == 3, ps.buses)
    slack_gen = only(filter(gen -> gen.bus == slack_bus, ps.gens))
    volt = Float64[]
    for bus in ps.buses
        append!(volt, (bus.v0m, bus.v0a))
    end
    sinj = zeros(length(volt))
    GradPower.compute_pinj!(sinj, volt, ps.network.ybus, length(ps.buses))
    local_pd = sum(load.pd for load in ps.loads if load.bus == slack_bus)

    @test slack_gen.psch ≈ sinj[2 * slack_bus - 1] + local_pd atol=1e-10
end
