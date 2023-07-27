@testset "Networktests 1" begin
    @testset "Case 9 tests" begin
        ps = create_case9()
        @testset "BaseMVA tests" begin
            @test ps.baseMVA == 100
        end

        @testset "Bus tests" begin
            @test length(ps.buses) == 9
            @test ps.buses[1].i == 1
            @test ps.buses[1].type == 3
            @test ps.buses[1].baseKV == 345
            @test ps.buses[9].i == 9
            @test ps.buses[9].type == 1
            @test ps.buses[9].baseKV == 345
        end

        @testset "Gen tests" begin
            @test length(ps.gens) == 3
            @test ps.gens[1].bus == 1
            @test ps.gens[1].psch == 0
            @test ps.gens[1].qsch == 0
            @test ps.gens[3].bus == 3
            @test ps.gens[3].psch == 0.85
            @test ps.gens[3].qsch == 0
        end

        @testset "Load tests" begin
            @test length(ps.loads) == 3
            @test ps.loads[1].bus == 5
            @test ps.loads[1].pd == 0.9
            @test ps.loads[1].qd == -0.3
            @test ps.loads[3].bus == 9
            @test ps.loads[3].pd == 1.25
            @test ps.loads[3].qd == -0.5
        end

        @testset "Branch tests" begin
            @test length(ps.branches) == 9
            @test ps.branches[1].fr == 1
            @test ps.branches[1].to == 4
            @test ps.branches[1].r == 0
            @test ps.branches[1].x == 0.0576
            @test ps.branches[9].fr == 9
            @test ps.branches[9].to == 4
            @test ps.branches[9].r == 0.01
            @test ps.branches[9].x == 0.085
        end

        @testset "Shunt tests" begin
            @test length(ps.shunts) == 0
        end
    end
end
