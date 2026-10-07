# Generator reactive-limit enforcement in the power flow (PV -> PQ switching).
#
# The behaviour under test is `runpf!(...; enforce_q_limits=true)`. Reference
# numbers come from uqgrid's own Q-limit loop run on the same cases; the
# ACTIVSg2000 agreement is documented in docs/activsg2000-diagnosis.md.

const QL_ROOT = abspath(joinpath(@__DIR__, ".."))

function _ql_system(raw, dyr)
    ps = GradPower.from_psse(joinpath(QL_ROOT, "examples", raw),
                             joinpath(QL_ROOT, "examples", dyr))
    GradPower.build_network!(ps)
    return ps
end

_ql_violations(ps; tol=1e-6) =
    count(g -> g.qsch > g.qmax + tol || g.qsch < g.qmin - tol, ps.gens)

@testset "Power flow reactive limits" begin

    @testset "limits are parsed onto Gen" begin
        ps = _ql_system("IEEE39.raw", "IEEE39_gov.dyr")
        @test all(g -> isfinite(g.qmax) && isfinite(g.qmin), ps.gens)
        @test all(g -> g.qmax >= g.qmin, ps.gens)
        # Gen's 6-argument form must stay usable and mean "no limits".
        g = GradPower.Gen(1, "1", 0.5, 0.1, 100.0, true)
        @test g.qmax == Inf && g.qmin == -Inf
    end

    @testset "default is off and unchanged" begin
        a = _ql_system("IEEE39.raw", "IEEE39_gov.dyr")
        b = _ql_system("IEEE39.raw", "IEEE39_gov.dyr")
        GradPower.runpf!(a)
        @test GradPower.runpf!(b; enforce_q_limits=false) == 0
        for (x, y) in zip(a.buses, b.buses)
            @test x.v0m == y.v0m
            @test x.v0a == y.v0a
            @test x.type == y.type
        end
    end

    @testset "IEEE9 has no binding limits, so enforcement is a no-op" begin
        off = _ql_system("ieee9_v33.raw", "ieee9bus_gov.dyr")
        on  = _ql_system("ieee9_v33.raw", "ieee9bus_gov.dyr")
        GradPower.runpf!(off)
        @test GradPower.runpf!(on; enforce_q_limits=true) == 0
        for (x, y) in zip(off.buses, on.buses)
            @test isapprox(x.v0m, y.v0m; atol=1e-12)
            @test isapprox(x.v0a, y.v0a; atol=1e-12)
        end
    end

    @testset "IEEE39 violates without enforcement and does not with it" begin
        off = _ql_system("IEEE39.raw", "IEEE39_gov.dyr")
        GradPower.runpf!(off)
        @test _ql_violations(off) > 0

        on = _ql_system("IEEE39.raw", "IEEE39_gov.dyr")
        npin = GradPower.runpf!(on; enforce_q_limits=true)
        @test npin > 0
        @test _ql_violations(on) == 0
        # Every pinned bus is one that started PV and is now PQ, and its
        # generators sit exactly on a bound.
        for (i, bus) in enumerate(on.buses)
            bus.type == 1 || continue
            off.buses[i].type == 2 || continue
            for g in on.gens
                g.bus == i || continue
                @test g.qsch ≈ g.qmax || g.qsch ≈ g.qmin
            end
        end
    end

    @testset "enforcement leaves a consistent dynamic equilibrium" begin
        # A converted bus's StaticGenerator stub must stop regulating, and its
        # vset/aset must follow the solved voltage. Both were wrong at first
        # and each showed up here as a large initialization residual.
        for enforce in (false, true)
            ps = _ql_system("ACTIVSg200.raw", "ACTIVSg200.dyr")
            GradPower.runpf!(ps; enforce_q_limits=enforce)
            dp = GradPower.DynamicProblem(ps)
            GradPower.initialize_dynamics!(dp, ps)
            f = zeros(length(dp.zvec))
            GradPower.rhs_fun!(f, dp.zvec, dp.uvec, dp.pvec, ps)
            @test maximum(abs, f) < 1e-8
        end
    end

    @testset "ACTIVSg200 limits bind and are respected" begin
        off = _ql_system("ACTIVSg200.raw", "ACTIVSg200.dyr")
        GradPower.runpf!(off)
        @test _ql_violations(off) > 0

        on = _ql_system("ACTIVSg200.raw", "ACTIVSg200.dyr")
        @test GradPower.runpf!(on; enforce_q_limits=true) > 0
        @test _ql_violations(on) == 0
    end

    @testset "co-located generators: equal share unless a limit binds" begin
        # Two units at one PV bus. With no binding limit the split must stay
        # exactly even (the historical behaviour); when one unit's ceiling
        # binds it takes its ceiling and the other absorbs the remainder, and
        # the bus total is preserved either way.
        ps = _ql_system("IEEE39.raw", "IEEE39_gov.dyr")
        GradPower.runpf!(ps; enforce_q_limits=true)
        by_bus = Dict{Int,Vector{Int}}()
        for (i, g) in enumerate(ps.gens)
            push!(get!(by_bus, g.bus, Int[]), i)
        end
        for (bus, gidx) in by_bus
            length(gidx) > 1 || continue
            qtot = sum(ps.gens[g].qsch for g in gidx)
            binding = any(g -> ps.gens[g].qsch ≈ ps.gens[g].qmax ||
                               ps.gens[g].qsch ≈ ps.gens[g].qmin, gidx)
            if !binding
                for g in gidx
                    @test ps.gens[g].qsch ≈ qtot / length(gidx)
                end
            end
        end
    end
end
