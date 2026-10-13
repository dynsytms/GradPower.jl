using Test
using GradPower

include(joinpath(@__DIR__, "regression", "common.jl"))

@testset "regression state manifest" begin
    raw = joinpath(@__DIR__, "testdata", "2bus.raw")
    dyr = joinpath(@__DIR__, "testdata", "2bus.dyr")
    ps = from_psse(raw, dyr)
    manifest = build_state_manifest(ps)
    @test [entry["index"] for entry in manifest] == collect(1:length(manifest))
    @test length(manifest) == ps.dynamic.diff_dim + ps.dynamic.alg_dim + 2length(ps.buses)
    @test any(entry -> entry["name"] == "e_qp", manifest)
    @test any(entry -> entry["name"] == "voltage_real", manifest)

    reference = [copy(entry) for entry in manifest]
    for entry in reference
        entry["index"] -= 1
    end
    rows = map_state_manifests(manifest, reference)
    @test length(rows) == length(manifest)
    @test map_state_manifests(manifest, manifest; reference_index_base=1) == rows

    t = [0.0, 0.1, 0.2]
    history = zeros(length(manifest), length(t))
    perturbed = copy(history)
    perturbed[1, 3] = 2e-3
    result = compare_trajectories(t, perturbed, t, history, rows; threshold=1e-3)
    @test result["max_abs_error"] == 2e-3
    @test result["worst_time"] == 0.2
    @test result["first_divergence_time"] == 0.2
end
