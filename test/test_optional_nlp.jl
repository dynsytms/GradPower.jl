@testset "optional NLP loading" begin
    project = dirname(Base.active_project())
    checks = joinpath(@__DIR__, "optional_nlp_checks.jl")
    for order in ("core", "gradpower_first", "nlp_first")
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project $checks $order`
        @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
    end
end
