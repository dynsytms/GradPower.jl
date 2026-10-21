@testset "GPU named results ($(nameof(typeof(ls))))" for ls in GPU_SOLVERS
    HAS_CUDA || return
    ps, dp = _result_setup()
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
    M = 2
    dt = 1 / 120
    tf = 6dt
    add_event!(ps, ContingencyEvent(4, 0.1, 2dt, 4dt))
    newgbl() = ext.GpuBatchedLayout(dp, ps, M; linear_solver=ls)
    for integrate in (ext.integrate_gpu_shared!, ext.integrate_gpu_cudss!,
                      ext.integrate_gpu_schur_cudss!)
        times, full = integrate(newgbl(), ps, tf)
        r = integrate(newgbl(), ps, tf; result=true, scenario_ids=[:a, :b])
        @test r isa SimulationResult
        @test r.times == times
        @test r.status == [:success, :success]
        @test r.scenario_ids == [:a, :b]
        @test getproperty.(r.events, :action) == [:fault_on, :fault_off]
        @test length(r.channels) == length(dp.zvec)
        for m in 1:M
            @test r.trajectories[m] ≈ full[m] atol=1e-7
        end
        c = first(filter(c -> c.model == :genrou && c.name == :w, r.channels))
        @test speed(r, 1; scenario=:b) == r.trajectories[2][c.index, :]
    end
    empty!(ps.dynamic.events)
    r = ext.integrate_gpu_shared_multi!(newgbl(), ps, tf, [4, 5], [0.1, 0.1], 2dt, 4dt;
                                        result=true, scenario_ids=[:a, :b], save_every=2)
    @test r isa SimulationResult
    @test length(r.events) == 4
    @test Set(e.details.scenario for e in r.events) == Set([:a, :b])
    @test Set(e.details.bus for e in r.events) == Set([ps.buses[4].i, ps.buses[5].i])
    @test size(r.trajectories[1], 2) == length(r.times) == 4
end
