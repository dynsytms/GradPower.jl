@testset "Continuation plotting selection" begin
    ps = cpf_controlled_bus()
    result = runcpf(ps;buses=[205])
    @test GradPower._plot_cpf_indices(result,nothing) == [1]
    @test GradPower._plot_cpf_indices(result,[205]) == [1]
    for buses in ([101], [205,205], Int[], [205.], [true], 205)
        @test_throws ArgumentError GradPower._plot_cpf_indices(result,buses)
    end
    empty_result = runcpf(ps;buses=Int[])
    @test_throws ArgumentError GradPower._plot_cpf_indices(empty_result,nothing)
    ps.loads[1].pd = 5.
    failed = runcpf(ps)
    @test isempty(failed.points)
    @test_throws ArgumentError GradPower._plot_cpf_indices(failed,nothing)
    if Base.get_extension(GradPower,:GradPowerMakieExt) === nothing
        @test_throws ArgumentError plot_cpf(result)
        @test_throws ArgumentError animate_cpf("unused.gif",result)
    end
end
