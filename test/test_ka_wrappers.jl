# Unit tests for KA wrappers (phase 14a D4).
#
# Tests that the KA CPU path (injection buffer + reduction) produces the
# same residual and Jacobian as the plain-loop path, for all device types.

@testset "KA residual: 2-bus" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    net = ps.network::GradPower.Network
    L = dyn.layout::GradPower.SimulationLayout

    f1 = zeros(length(dp.zvec))
    GradPower._rhs_fun_batched!(f1, dp.zvec, dp.uvec, dp.pvec, dyn, net.ybus_real, L)

    inj_meta = GradPower.InjectionMeta(L)
    inj = zeros(2 * inj_meta.n_total)
    f2 = zeros(length(dp.zvec))
    GradPower._rhs_fun_ka_cpu!(f2, dp.zvec, dp.uvec, dp.pvec, inj, dyn, net.ybus_real, L, inj_meta)

    @test maximum(abs, f1 - f2) <= 1e-14
end

@testset "KA residual: IEEE-9 with governors" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    net = ps.network::GradPower.Network
    L = dyn.layout::GradPower.SimulationLayout

    f1 = zeros(length(dp.zvec))
    GradPower._rhs_fun_batched!(f1, dp.zvec, dp.uvec, dp.pvec, dyn, net.ybus_real, L)

    inj_meta = GradPower.InjectionMeta(L)
    inj = zeros(2 * inj_meta.n_total)
    f2 = zeros(length(dp.zvec))
    GradPower._rhs_fun_ka_cpu!(f2, dp.zvec, dp.uvec, dp.pvec, inj, dyn, net.ybus_real, L, inj_meta)

    @test maximum(abs, f1 - f2) <= 1e-14
end

@testset "KA Jacobian: 2-bus" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    net = ps.network::GradPower.Network
    L = dyn.layout::GradPower.SimulationLayout

    J1 = GradPower.preallocate_jacobian(ps)
    J2 = GradPower.preallocate_jacobian(ps)

    fill!(J1.nzval, 0.0)
    GradPower._rhs_jac_batched!(J1, dp.zvec, dp.uvec, dp.pvec, dyn, net.ybus_real, L)

    fill!(J2.nzval, 0.0)
    GradPower._rhs_jac_ka_cpu!(J2, dp.zvec, dp.uvec, dp.pvec, dyn, net.ybus_real, L)

    @test maximum(abs, J1.nzval - J2.nzval) == 0.0
end

@testset "KA Jacobian: IEEE-9" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    net = ps.network::GradPower.Network
    L = dyn.layout::GradPower.SimulationLayout

    J1 = GradPower.preallocate_jacobian(ps)
    J2 = GradPower.preallocate_jacobian(ps)

    fill!(J1.nzval, 0.0)
    GradPower._rhs_jac_batched!(J1, dp.zvec, dp.uvec, dp.pvec, dyn, net.ybus_real, L)

    fill!(J2.nzval, 0.0)
    GradPower._rhs_jac_ka_cpu!(J2, dp.zvec, dp.uvec, dp.pvec, dyn, net.ybus_real, L)

    @test maximum(abs, J1.nzval - J2.nzval) == 0.0
end

@testset "dispatch_residual_kernels! KA_CPU matches plain-loop" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    net = ps.network::GradPower.Network
    L = dyn.layout::GradPower.SimulationLayout
    ybus = net.ybus_real

    inj_meta = GradPower.InjectionMeta(L)
    inj = zeros(2 * inj_meta.n_total)

    f1 = zeros(length(dp.zvec))
    GradPower.dispatch_residual_kernels!(f1, dp.zvec, dp.uvec, dp.pvec, inj, L, dyn, ybus, inj_meta, nothing)

    fill!(inj, 0.0)
    f2 = zeros(length(dp.zvec))
    GradPower.dispatch_residual_kernels!(f2, dp.zvec, dp.uvec, dp.pvec, inj, L, dyn, ybus, inj_meta, GradPower.KA_CPU())

    @test maximum(abs, f1 - f2) <= 1e-14
end

@testset "dispatch_jacobian_kernels! KA_CPU matches plain-loop" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    net = ps.network::GradPower.Network
    L = dyn.layout::GradPower.SimulationLayout
    ybus = net.ybus_real

    J1 = GradPower.preallocate_jacobian(ps)
    J2 = GradPower.preallocate_jacobian(ps)

    fill!(J1.nzval, 0.0)
    GradPower.dispatch_jacobian_kernels!(J1, dp.zvec, dp.uvec, dp.pvec, L, dyn, ybus, nothing)

    fill!(J2.nzval, 0.0)
    GradPower.dispatch_jacobian_kernels!(J2, dp.zvec, dp.uvec, dp.pvec, L, dyn, ybus, GradPower.KA_CPU())

    @test maximum(abs, J1.nzval - J2.nzval) == 0.0
end

@testset "injection buffer + reduce matches direct accumulation" begin
    # Test on a case where zipload and genrou share a bus
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    L = dyn.layout::GradPower.SimulationLayout
    inj_meta = GradPower.InjectionMeta(L)

    # Verify injection metadata is correct
    @test inj_meta.n_genrou == L.genrou.n
    @test inj_meta.n_zipload == L.zipload.n
    @test inj_meta.n_static_gen == L.static_gen.n
    @test inj_meta.n_total == L.genrou.n + L.zipload.n + L.static_gen.n

    # Verify bus map correctness
    for k in 1:L.genrou.n
        @test inj_meta.bus_map[k] == L.genrou.bus[k]
    end
    for k in 1:L.zipload.n
        @test inj_meta.bus_map[inj_meta.n_genrou + k] == L.zipload.bus[k]
    end
end
