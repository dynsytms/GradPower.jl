# Unit tests for phase 14c — lockstep GPU pipeline.
# All GPU tests guarded by CUDA.functional() — skip gracefully on CPU-only machines.

const HAS_CUDA_14c = try
    @eval using CUDA
    @eval using CUDSS
    @eval using CUDA.CUSPARSE
    CUDA.functional()
catch
    false
end

if !HAS_CUDA_14c
    @info "No functional CUDA device — skipping lockstep GPU tests"
end

@testset "lockstep" begin

@testset "Controller KA residual kernels match CPU (≤ 1e-14)" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    L = dyn.layout
    f_ref = zeros(length(dp.zvec))
    GradPower._rhs_fun_batched!(f_ref, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    # KA CPU path
    inj_meta = GradPower.InjectionMeta(L)
    inj = zeros(2 * inj_meta.n_total)
    f_ka = zeros(length(dp.zvec))
    GradPower._rhs_fun_ka_cpu!(f_ka, dp.zvec, dp.uvec, dp.pvec, inj,
                                dyn, ps.network.ybus_real, L, inj_meta)

    @test maximum(abs, f_ka - f_ref) <= 1e-14
end

@testset "Controller KA Jacobian kernels match CPU (≤ 1e-14)" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    L = dyn.layout

    J_ref = GradPower.preallocate_jacobian(ps)
    GradPower._rhs_jac_batched!(J_ref, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    J_ka = GradPower.preallocate_jacobian(ps)
    GradPower._rhs_jac_ka_cpu!(J_ka, dp.zvec, dp.uvec, dp.pvec,
                                dyn, ps.network.ybus_real, L)

    @test maximum(abs, J_ka.nzval - J_ref.nzval) <= 1e-14
end

@testset "GPU Jacobian nzval matches CPU (≤ 1e-14)" begin
    HAS_CUDA_14c || return

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    L = dyn.layout

    gbl = ext.GpuBatchedLayout(dp, ps, 1)
    ext._jacobian_all_scenarios_gpu!(gbl, dyn, L)
    J_gpu = Array(gbl.J_nzval)

    J_ref = GradPower.preallocate_jacobian(ps)
    GradPower._rhs_jac_batched!(J_ref, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    @test maximum(abs, J_gpu[1,:] - J_ref.nzval) <= 1e-14
end

@testset "2D batched kernels match per-scenario loop (≤ 1e-14)" begin
    HAS_CUDA_14c || return

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    L = dyn.layout
    M = 4

    # GPU batched residual
    gbl = ext.GpuBatchedLayout(dp, ps, M)
    ext._residual_all_scenarios_gpu!(gbl, dyn, L)
    f_gpu = Array(gbl.f)

    # CPU reference (single scenario)
    f_ref = zeros(length(dp.zvec))
    GradPower._rhs_fun_batched!(f_ref, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    # All scenarios should match
    for m in 1:M
        @test maximum(abs, f_gpu[m,:] - f_ref) <= 1e-14
    end
end

@testset "GPU convergence check (D7)" begin
    HAS_CUDA_14c || return

    # Verify CUDA.mapreduce(abs, max, ...) gives same result as CPU
    f = CUDA.rand(Float64, 4, 10) .- 0.5
    gpu_max = CUDA.mapreduce(abs, max, f)
    cpu_max = maximum(abs, Array(f))
    @test abs(gpu_max - cpu_max) <= 1e-15
end

@testset "GPU single-scenario trajectory matches CPU (≤ 1e-6)" begin
    HAS_CUDA_14c || return

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end

    # CPU reference
    dp1 = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp1, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))
    tvec_ref, traj_ref = GradPower.integrate!(dp1, ps, 0.5; dt=1.0/120.0,
                                                solver=:monolithic, newton_tol=1e-10)

    for ev in ps.dynamic.events; GradPower.deactivate!(ev); end
    empty!(ps.dynamic.events)

    # GPU M=1
    dp2 = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp2, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))

    gbl = ext.GpuBatchedLayout(dp2, ps, 1)
    tvec_gpu, trajs_gpu = ext.integrate_gpu_cudss!(gbl, ps, 0.5; dt=1.0/120.0, newton_tol=1e-10)

    @test maximum(abs, trajs_gpu[1] - traj_ref) <= 1e-6
end

@testset "Lockstep mask: converged scenario gets zero update" begin
    # This is a CPU-only test of the lockstep concept.
    # When a scenario has converged (residual < tol), its state update
    # should be zero (or very small).
    # Test: if we solve with dt=0 and zold=z, residual=0, dx should be 0.
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    L = dyn.layout

    f = zeros(length(dp.zvec))
    GradPower.beuler_batched!(f, dp.zvec, dp.zvec, dp.uvec, dp.pvec,
                               dyn, ps.network.ybus_real, L, dyn.diff_dim, 0.0)

    # At z = z0 with dt=0, residual should be ~0
    @test maximum(abs, f) <= 1e-9
end

end # @testset "lockstep"
