using Test
using GradPower

@testset "MATPOWER mat_to_grad" begin
    mpc = Dict{Any,Any}(
        "baseMVA" => 100.0,
        "bus" => [
            Dict("bus_i"=>1.0, "type"=>3.0, "Pd"=>0.0, "Qd"=>0.0,
                 "Gs"=>0.0, "Bs"=>0.0, "area"=>1.0, "Vm"=>1.0, "Va"=>0.0,
                 "baseKV"=>100.0, "zone"=>1.0, "Vmax"=>1.1, "Vmin"=>0.9),
            Dict("bus_i"=>2.0, "type"=>1.0, "Pd"=>50.0, "Qd"=>10.0,
                 "Gs"=>0.0, "Bs"=>0.0, "area"=>1.0, "Vm"=>1.0, "Va"=>0.0,
                 "baseKV"=>100.0, "zone"=>1.0, "Vmax"=>1.1, "Vmin"=>0.9),
        ],
        "gen" => [
            Dict("bus"=>1.0, "Pg"=>50.0, "Qg"=>10.0, "Qmax"=>100.0, "Qmin"=>-100.0,
                 "Vg"=>1.0, "mBase"=>100.0, "status"=>1.0,
                 "Pmax"=>200.0, "Pmin"=>0.0, "Pc1"=>0.0, "Pc2"=>0.0,
                 "Qc1min"=>0.0, "Qc1max"=>0.0, "Qc2min"=>0.0, "Qc2max"=>0.0,
                 "ramp_agc"=>0.0, "ramp_10"=>0.0, "ramp_30"=>0.0, "ramp_q"=>0.0, "apf"=>0.0),
            Dict("bus"=>2.0, "Pg"=>0.0, "Qg"=>0.0, "Qmax"=>0.0, "Qmin"=>0.0,
                 "Vg"=>1.0, "mBase"=>100.0, "status"=>0.0,
                 "Pmax"=>0.0, "Pmin"=>0.0, "Pc1"=>0.0, "Pc2"=>0.0,
                 "Qc1min"=>0.0, "Qc1max"=>0.0, "Qc2min"=>0.0, "Qc2max"=>0.0,
                 "ramp_agc"=>0.0, "ramp_10"=>0.0, "ramp_30"=>0.0, "ramp_q"=>0.0, "apf"=>0.0),
        ],
        "branch" => [
            Dict("fbus"=>1.0, "tbus"=>2.0, "r"=>0.01, "x"=>0.1, "b"=>0.0,
                 "rateA"=>0.0, "rateB"=>0.0, "rateC"=>0.0,
                 "ratio"=>1.0, "angle"=>0.0, "status"=>1.0,
                 "angmin"=>-360.0, "angmax"=>360.0),
        ],
    )

    ps = GradPower.mat_to_grad(mpc)

    @test length(ps.gens) == 2
    @test ps.gens[1].status === true
    @test ps.gens[2].status === false
    @test ps.gens[1].bus == 1
    @test ps.gens[2].bus == 2
    @test ps.gens[1].psch ≈ 0.5    # 50/100
    @test ps.gens[1].qsch ≈ 0.1    # 10/100
    @test ps.gens[1].mbase ≈ 100.0
end
