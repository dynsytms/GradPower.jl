@testset "Dynamics 1" begin
    @testset "Case 2 bus dynamics" begin
            # Test case 2 bus dynamics
            raw_file = "testdata/2bus.raw"
            dyr_file = "testdata/2bus.dyr"
            raw = GradPower.read_psse_raw(raw_file)
            sys = GradPower.raw_to_grad(raw)
            psd = GradPower.PowerSystemDynamics(dyr_file)
            GradPower.set_dynamics!(sys, psd)

            # power flow
            GradPower.build_network!(sys)
            GradPower.runpf!(sys, verbose=false);

            # dynamic simulation
            tfinal = 1.0
            dprob = GradPower.DynamicProblem(sys)
            GradPower.initialize_dynamics!(dprob, sys)

            # add event
            event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))
            tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
            result = traj[:, end]

            @test isapprox(result[1], 1.0276157414718632, rtol=1e-2)
            @test isapprox(result[4], -0.6178842140068552, rtol=1e-2)
            @test isapprox(result[10], 0.9337400271910671, rtol=1e-2)
            @test isapprox(result[14], -0.9344382816186252, rtol=1e-2)
    end
end
