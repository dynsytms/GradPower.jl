# The metrics on synthetic trajectories with known answers.
@testset "metrics" begin
    t = collect(0:1/240:2.0)
    nt = length(t)
    mk(; delta, omega=zeros(nt, 2), efd=ones(nt, 2), pm=fill(0.5, nt, 2), vm=ones(nt, 3), t=t) =
        (t=t, bus_i=[1, 2, 3], vm_bus=[1, 2, 3], pf_vm=[1.0, 1.02, 0.99], pf_va=[0.0, 0.1, -0.05],
         gens=[(1, "1"), (2, "1")], delta=delta, omega=omega, efd=efd, pm=pm, vm=vm)
    swing = [0.0 * t (t .> 0.2) .* deg2rad(10) .* sin.(2pi .* (t .- 0.2))]   # machine 2: 10 degree swing
    ref = mk(delta=swing)
    trip = Dict("kind" => "trip", "t_on" => 0.2)

    m = PSSEReference.compare_runs(ref, ref, trip)
    @test m["angle_pct"] == 0 && m["v_err"] == 0 && m["pf_dvm"] == 0
    @test m["swing_ref_deg"] ≈ 10 atol = 1e-3
    @test isnan(m["efd_pct"]) && isnan(m["speed_pct"])            # no reference excursion

    # 0.5 degree extra on machine 2: 5 % of the swing
    m = PSSEReference.compare_runs(mk(delta=swing .+ [0.0 * t (t .> 0.2) .* deg2rad(0.5)]), ref, trip)
    @test m["angle_pct"] ≈ 5 atol = 1e-6

    # Efd: a 0.2 pu step in the reference, GradPower 0.01 pu low after it -> 5 %;
    # a spike inside the 0.05 s window around the switching instant is left out
    efd_r = ones(nt, 2) .+ 0.2 .* (t .> 0.3)
    efd_g = efd_r .- 0.01 .* (t .> 0.3) .+ 0.5 .* (abs.(t .- 0.2) .< 0.04)
    m = PSSEReference.compare_runs(mk(delta=swing, efd=efd_g), mk(delta=swing, efd=efd_r), trip)
    @test m["efd_pct"] ≈ 5 atol = 1e-6

    # PSS/E writes two rows at a switching instant; the grid must still interpolate
    k = findfirst(>=(0.2), t)
    t2 = vcat(t[1:k], t[k:end])
    dup(X) = vcat(X[1:k, :], X[k:end, :])
    m = PSSEReference.compare_runs(ref, mk(delta=dup(swing), efd=dup(ones(nt, 2)), omega=dup(zeros(nt, 2)),
                                           pm=dup(fill(0.5, nt, 2)), vm=dup(ones(nt, 3)), t=t2), trip)
    @test m["angle_pct"] < 1e-6

    # power-flow angles are compared relative to the first bus, wrapped
    shifted = merge(ref, (pf_va=ref.pf_va .+ [0.3, 0.3 + 2pi, 0.3],))
    @test PSSEReference.compare_runs(shifted, ref, trip)["pf_dva_deg"] < 1e-9
end
