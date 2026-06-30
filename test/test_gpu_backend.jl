# Unit tests for GPU backend (phase 14b D5).
# All GPU tests guarded by CUDA.functional() — skip gracefully on CPU-only machines.
# If CUDA/CUDSS packages are not available, all tests are skipped.

const HAS_CUDA = try
    @eval using CUDA
    @eval using CUDSS
    @eval using CUDA.CUSPARSE
    CUDA.functional()
catch
    false
end

if !HAS_CUDA
    @info "No functional CUDA device or CUDA packages not available — skipping GPU backend tests"
end

@testset "gpu_backend" begin

@testset "CuArray GpuBatchedLayout construction" begin
    HAS_CUDA || return

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
    M = 4
    gbl = ext.GpuBatchedLayout(dp, ps, M)

    @test gbl.M == M
    @test gbl.sys_dim == length(dp.zvec)
    @test gbl.z isa CuMatrix{Float64}
    @test size(gbl.z) == (M, gbl.sys_dim)
    @test gbl.p isa CuMatrix{Float64}
    @test gbl.u isa CuMatrix{Float64}
    @test gbl.f isa CuMatrix{Float64}

    # Each scenario matches single-scenario IC
    z_cpu = Array(gbl.z)
    p_cpu = Array(gbl.p)
    u_cpu = Array(gbl.u)
    for m in 1:M
        @test z_cpu[m, :] == dp.zvec
        @test p_cpu[m, :] == dp.pvec
        @test u_cpu[m, :] == dp.uvec
    end
end

@testset "cuDSS preconditioner matches KLU" begin
    HAS_CUDA || return

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)

    Y = ps.network.ybus_real
    n = size(Y, 1)

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    # Use a well-conditioned RHS: b = Y * x_true where x_true is random
    x_true = randn(n)
    b = Y * x_true

    # KLU reference
    klu_fact = GradPower.KLU.klu(Y)
    y_cpu = similar(b)
    ldiv!(y_cpu, klu_fact, b)

    # cuDSS
    prec = ext.CuDSSPreconditioner(Y)
    b_gpu = CuVector(b)
    y_gpu = CUDA.zeros(Float64, n)
    ldiv!(y_gpu, prec, b_gpu)
    y_from_gpu = Array(y_gpu)

    # Both should have small backward error (relative residual).
    residual_klu = norm(Y * y_cpu - b) / norm(b)
    residual_gpu = norm(Y * y_from_gpu - b) / norm(b)

    @test residual_klu <= 1e-10
    @test residual_gpu <= 1e-10
end

@testset "Batched cuBLAS LU matches CPU" begin
    HAS_CUDA || return

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    w_k = 5
    n_clusters = 3
    M = 4
    batch = n_clusters * M

    # Random diagonally-dominant dense systems
    A_cpu = [randn(w_k, w_k) + 10.0 * I for _ in 1:batch]
    B_cpu = [randn(w_k, 2) for _ in 1:batch]

    # CPU reference: LU + solve
    x_ref = similar.(B_cpu)
    for b in 1:batch
        F = lu(A_cpu[b])
        x_ref[b] = F \ B_cpu[b]
    end

    # GPU path
    glu = ext.GpuBatchedLU(w_k, n_clusters, M)
    # Pack A and B into 3D arrays
    A_packed = zeros(Float64, w_k, w_k, batch)
    for b in 1:batch; A_packed[:, :, b] .= A_cpu[b]; end
    copyto!(glu.A_packed, CuArray(A_packed))

    B_packed = zeros(Float64, w_k, 2, batch)
    for b in 1:batch; B_packed[:, :, b] .= B_cpu[b]; end
    copyto!(glu.B_packed, CuArray(B_packed))

    ext.gpu_batched_lu_factor!(glu)
    ext.gpu_batched_lu_solve!(glu)

    B_result = Array(glu.B_packed)
    max_err = 0.0
    for b in 1:batch
        err = maximum(abs, B_result[:, :, b] - x_ref[b])
        max_err = max(max_err, err)
    end
    @test max_err <= 1e-12
end

@testset "GPU Schur operator matches CPU" begin
    HAS_CUDA || return

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

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    ct = dyn.clusters::GradPower.ClusterTable
    net_ptr = dyn.diff_dim + dyn.alg_dim
    dt = 1.0 / 120.0

    # Build SchurWorkspace and assemble S
    sw = GradPower.SchurWorkspace(ps)
    J0 = GradPower.preallocate_jacobian(ps)
    GradPower.beuler_jac_batched!(J0, dp.zvec, dp.uvec, dp.pvec,
                                   dyn, ps.network.ybus_real,
                                   dyn.layout::GradPower.SimulationLayout,
                                   dyn.diff_dim, dt)
    GradPower.assemble_schur!(sw, J0, ct, net_ptr)

    # GPU Schur operator
    op = ext.GpuSchurOperator(length(sw.reduced_idx), sw)
    n_red = length(sw.reduced_idx)

    # Random test vector
    x = randn(n_red)
    y_op = zeros(n_red)
    y_ref = zeros(n_red)

    mul!(y_op, op, x)
    mul!(y_ref, sw.S, x)

    diff = maximum(abs, y_op - y_ref)
    @test diff <= 1e-12
end

@testset "KA CUDA kernel dispatch — single-scenario residual" begin
    HAS_CUDA || return

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

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    L = dyn.layout::GradPower.SimulationLayout

    # CPU reference
    f_ref = zeros(length(dp.zvec))
    GradPower._rhs_fun_batched!(f_ref, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    # GPU path: build GpuBatchedLayout with M=1, evaluate residual
    gbl = ext.GpuBatchedLayout(dp, ps, 1)
    ext._residual_all_scenarios_gpu!(gbl, dyn, L)
    f_gpu = Array(gbl.f)

    diff = maximum(abs, f_gpu[1, :] - f_ref)
    @test diff <= 1e-12
end

end # @testset "gpu_backend"
