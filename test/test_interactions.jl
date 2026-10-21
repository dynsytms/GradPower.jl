using Test
using GradPower

function _interaction_setup(dyr="2bus_TGOV1.dyr"; raw="2bus.raw")
    ex = joinpath(@__DIR__, "..", "examples")
    ps = from_psse(joinpath(ex, raw), joinpath(ex, dyr))
    build_network!(ps)
    runpf!(ps)
    dp = DynamicProblem(ps)
    initialize_dynamics!(dp, ps)
    return ps, dp
end

function _interaction_residual(ps, dp, z=dp.zvec)
    f = zeros(length(z))
    GradPower.beuler_batched!(f, z, z, dp.uvec, dp.pvec, ps.dynamic,
        ps.network.ybus_real, ps.dynamic.layout, ps.dynamic.diff_dim, 0.01)
    return f
end

@testset "Named parameter interactions" begin
    fixtures = [
        ("2bus.dyr", :genrou, :D),
        ("2bus_IEESGO.dyr", :ieesgo, :K1),
        ("2bus_TGOV1.dyr", :tgov1, :DT),
        ("2bus_SEXS.dyr", :sexs, :K),
        ("2bus_ESDC1A.dyr", :esdc1a, :sat_a),
        ("2bus_IEEEST.dyr", :ieeest, :KS),
    ]
    for (file, model, parameter) in fixtures
        ps, dp = _interaction_setup(file)
        i = first(find_devices(ps; model))
        dev = ps.dynamic.devices[i]
        bus = ps.buses[ps.dynamic.map.bus[i]].i
        id = GradPower._result_device_id(ps, dev.dtype)
        for name in get_param_names(dev.dtype)
            @test get_parameter(ps, i, name) == get_parameter(dp, ps, i, name)
            GradPower._parameter_readonly(dev.dtype, Symbol(name)) && continue
            unchanged = get_parameter(ps, i, name)
            @test set_parameter!(dp, ps, i, name, unchanged) == unchanged
            column, row = GradPower._parameter_snapshot(ps, dev, Symbol(name))
            column === nothing || @test column[row] == unchanged
        end
        old = get_parameter(dp, ps, model, bus, id, parameter)
        value = old + 0.01
        @test set_parameter!(dp, ps, model, bus, " '$id' ", parameter, value) == value
        @test get_parameter(ps, model, bus, id, parameter) == value
        @test get_parameter(dp, ps, i, parameter) == value
        column, row = GradPower._parameter_snapshot(ps, dev, parameter)
        @test column[row] == value
        initialize_dynamics!(dp, ps)
        @test get_parameter(dp, ps, i, parameter) == value
        @test column[row] == value
        @test maximum(abs, _interaction_residual(ps, dp)) < 1e-7
        @test_throws ArgumentError set_parameter!(dp, ps, i, :unknown, 1.0)
        @test_throws ArgumentError set_parameter!(dp, ps, i, parameter, NaN)
        @test get_parameter(dp, ps, i, parameter) == value
        @test column[row] == value
        @test_throws ArgumentError set_parameter!(dp, ps, i, :bus, 2)
    end

    @testset "Configuration before initialization" begin
        ps, _ = _interaction_setup()
        dp = DynamicProblem(ps)
        i = first(find_devices(ps; model=:genrou))
        set_parameter!(dp, ps, i, :D, 2.5)
        initialize_dynamics!(dp, ps)
        @test get_parameter(dp, ps, i, :D) == 2.5
        @test maximum(abs, _interaction_residual(ps, dp)) < 1e-7
    end

    @testset "Edits reach residuals and snapshots" begin
        ps, dp = _interaction_setup()
        i = first(find_devices(ps; model=:genrou))
        dev = ps.dynamic.devices[i]
        bus = ps.buses[ps.dynamic.map.bus[i]].i
        id = GradPower._result_device_id(ps, dev.dtype)
        z = copy(dp.zvec)
        z[state_index(ps, :genrou, bus, id, :w)] = 0.02
        before = _interaction_residual(ps, dp, z)
        set_parameter!(dp, ps, i, :D, 3.0)
        after = _interaction_residual(ps, dp, z)
        @test maximum(abs, before - after) > 1e-8
        @test ps.dynamic.layout.genrou.D[1] == 3.0

        li = first(find_devices(ps; model=:zipload))
        alpha = Symbol("\u03b1")
        voltage = ps.dynamic.diff_dim + ps.dynamic.alg_dim +
                  2 * (ps.dynamic.map.bus[li] - 1) + 1
        z[voltage] *= 0.8
        before = _interaction_residual(ps, dp, z)
        set_parameter!(dp, ps, li, alpha, 0.0)
        after = _interaction_residual(ps, dp, z)
        @test maximum(abs, before - after) > 1e-6
        initialize_dynamics!(dp, ps)
        @test get_parameter(dp, ps, li, alpha) == 0.0
    end

    @testset "Derived and structural parameters are read-only" begin
        for (file, model, names) in [
            ("2bus_IEESGO.dyr", :ieesgo, [:pref]),
            ("2bus_TGOV1.dyr", :tgov1, [:pref]),
            ("2bus_SEXS.dyr", :sexs, [:vref]),
            ("2bus_ESDC1A.dyr", :esdc1a, [:vref]),
            ("2bus_IEEEST.dyr", :ieeest, [:MODE, :BUSR]),
            ("2bus.dyr", :zipload, [:pinj, :qinj, :v0mag, :yreal, :yimag]),
        ]
            ps, dp = _interaction_setup(file)
            i = first(find_devices(ps; model))
            for name in names
                old = get_parameter(dp, ps, i, name)
                @test_throws ArgumentError set_parameter!(dp, ps, i, name, old + 1)
                @test get_parameter(dp, ps, i, name) == old
            end
        end
        ps, dp = _interaction_setup()
        i = first(find_devices(ps; model=:tgov1))
        old = get_parameter(dp, ps, i, :VMIN)
        @test_throws ArgumentError set_parameter!(dp, ps, i, :VMIN,
                                                get_parameter(ps, i, :VMAX) + 1)
        @test get_parameter(dp, ps, i, :VMIN) == old
        bad = DynamicProblem(dp.zvec, dp.uvec, Float64[])
        @test_throws DimensionMismatch set_parameter!(bad, ps, i, :DT, 1.0)
        for name in (:R, :T1, :T3)
            old = get_parameter(ps, i, name)
            old_p = copy(dp.pvec)
            column, row = GradPower._parameter_snapshot(ps, ps.dynamic.devices[i], name)
            for zero in (0.0, -0.0)
                @test_throws ArgumentError set_parameter!(dp, ps, i, name, zero)
                @test get_parameter(ps, i, name) == old
                @test dp.pvec == old_p
                @test column[row] == old
            end
        end
        # A static-generator stub's parameters are all derived from PF.
        ps = deepcopy(ps)
        ps.dynamic = nothing
        set_dynamics!(ps, PowerSystemDynamics())
        dp = DynamicProblem(ps)
        initialize_dynamics!(dp, ps)
        i = first(find_devices(ps; model=:static_gen))
        for name in get_param_names(ps.dynamic.devices[i].dtype)
            @test_throws ArgumentError set_parameter!(dp, ps, i, name, 1.0)
        end
    end

    @testset "Unenforced equal limits do not block other edits" begin
        for (file, model, lower, upper, gain) in [
            ("2bus_IEEEST.dyr", :ieeest, :LSMIN, :LSMAX, :KS),
            ("2bus_IEESGO.dyr", :ieesgo, :pmin, :pmax, :K1),
            ("2bus_SEXS.dyr", :sexs, :EMIN, :EMAX, :K),
        ]
            ps, dp = _interaction_setup(file)
            i = first(find_devices(ps; model))
            set_parameter!(dp, ps, i, lower, 0.0)
            set_parameter!(dp, ps, i, upper, 0.0)
            value = get_parameter(ps, i, gain) * 1.01
            @test set_parameter!(dp, ps, i, gain, value) == value
            @test get_parameter(ps, i, gain) == value
            @test get_parameter(dp, ps, i, gain) == value
            initialize_dynamics!(dp, ps)
            @test get_parameter(dp, ps, i, gain) == value
            @test maximum(abs, _interaction_residual(ps, dp)) < 1e-7
        end
    end
end

@testset "Pre-power-flow schedule edits" begin
    ps, _ = _interaction_setup("ieee9bus_gov.dyr"; raw="ieee9_v33.raw")
    li = first(find_devices(ps; model=:zipload))
    load_index = ps.dynamic.map.load[li]
    load = ps.loads[load_index]
    bus = ps.buses[load.bus].i
    p, q = 1.1load.pd, 1.1load.qd
    @test set_load!(ps, bus, " '$(GradPower._normalize_id(load.id))' "; pd=p, qd=q) === load
    @test (load.pd, load.qd) == (p, q)
    @test get_parameter(ps, li, :pinj) == p
    @test get_parameter(ps, li, :qinj) == q
    col, row = GradPower._parameter_snapshot(ps, ps.dynamic.devices[li], :pinj)
    @test col[row] == p
    @test_throws ArgumentError set_load!(ps, bus, load.id; pd=2p, qd=NaN)
    @test (load.pd, load.qd) == (p, q)
    @test_throws ArgumentError set_load!(ps, bus, "missing"; pd=p)
    @test_throws ArgumentError set_load!(ps, -1, load.id; pd=p)

    gen = ps.gens[2]
    gb = ps.buses[gen.bus].i
    pnew = 1.05gen.psch
    @test set_generator!(ps, gb, gen.id; psch=pnew, qmin=-2, qmax=2) === gen
    @test (gen.psch, gen.qmin, gen.qmax) == (pnew, -2.0, 2.0)
    @test_throws ArgumentError set_generator!(ps, gb, gen.id; psch=NaN)
    @test_throws ArgumentError set_generator!(ps, gb, gen.id; psch=2pnew, qmin=3)
    @test gen.psch == pnew
    @test_throws ArgumentError set_generator!(ps, gb, gen.id; qmin=Inf, qmax=Inf)
    @test_throws ArgumentError set_generator!(ps, gb, gen.id; qmin=-Inf, qmax=-Inf)
    set_generator!(ps, gb, gen.id; qmin=-Inf, qmax=Inf)
    @test (gen.qmin, gen.qmax) == (-Inf, Inf)

    runpf!(ps)
    dp = DynamicProblem(ps)
    initialize_dynamics!(dp, ps)
    @test get_parameter(dp, ps, li, :pinj) == p
    @test get_parameter(dp, ps, li, :qinj) == q
    @test maximum(abs, _interaction_residual(ps, dp)) < 1e-7
    times, trajectory = integrate!(dp, ps, 0.025)
    @test last(times) == 0.025
    @test all(isfinite, trajectory)

    # Static-only systems support the same editing API.
    ps.dynamic = nothing
    @test set_load!(ps, load.bus, load.id; pd=p / 2, numbering=:internal).pd == p / 2
    @test set_generator!(ps, gen.bus, gen.id; qsch=0.1, numbering=:internal).qsch == 0.1
end
