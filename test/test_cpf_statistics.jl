@testset "Continuation work counters" begin
    ps=cpf_controlled_bus()
    baseline=runcpf(ps;buses=[205])
    stats=ContinuationStats()
    logged=runcpf(ps;buses=[205],stats)
    @test logged.status == baseline.status
    @test [p.lambda for p in logged.points] == [p.lambda for p in baseline.points]
    @test [(e.bus,e.kind,e.point.lambda) for e in logged.events] ==
          [(e.bus,e.kind,e.point.lambda) for e in baseline.events]
    @test stats.setups == 1
    @test stats.residuals > length(logged.points)
    @test stats.jacobians > 0
    @test stats.factorizations == stats.solves
    @test stats.jacobians == stats.factorizations+1 # fold check evaluates J once
    @test stats.fold_checks == 1
    @test all(getproperty(stats,k)>0 for k in
              (:setup_ns,:residual_ns,:jacobian_ns,:factorization_ns,:solve_ns,:fold_ns))
    counts=(stats.residuals,stats.factorizations)
    runcpf(ps;buses=[205],stats)
    @test stats.setups == 2
    @test (stats.residuals,stats.factorizations) == 2 .* counts
    empty_log=ContinuationStats()
    runcpf(cpf_two_bus();limits=:none,max_steps=0,stats=empty_log)
    @test empty_log.setups == 1
    @test empty_log.fold_checks == 0
    @test empty_log.fold_ns == 0
    @test empty_log.residuals > 0
end
