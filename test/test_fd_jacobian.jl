# Finite-difference Jacobian validation for all device kernels.
#
# For each device type, the analytical Jacobian (from xxx_jacobian_batch!)
# is compared against a central-difference FD Jacobian of the residual.
# This is the ground-truth check — parity tests only verify that two
# implementations agree, not that either is correct.

const EX = joinpath(@__DIR__, "..", "examples")

# Helper: build system, init, return (ps, dp, L, diff_dim, net_ptr, sysdim)
function _setup(raw, dyr)
    ps = from_psse(joinpath(EX, raw), joinpath(EX, dyr))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    L = ps.dynamic.layout
    dd = ps.dynamic.diff_dim
    np = dd + ps.dynamic.alg_dim
    n  = np + 2 * length(ps.buses)
    return ps, dp, L, dd, np, n
end

# Helper: max |J_a - J_fd| on specific rows
function _max_row_err(J_a, J_fd, rows)
    maxe = 0.0
    for r in rows, c in 1:size(J_fd, 2)
        maxe = max(maxe, abs(J_a[r, c] - J_fd[r, c]))
    end
    return maxe
end

# -----------------------------------------------------------------------
# Per-device FD Jacobian tests
# -----------------------------------------------------------------------

@testset "FD Jacobian: GENROU" begin
    ps, dp, L, dd, np, n = _setup("2bus.raw", "2bus.dyr")
    tab = L.genrou
    @test tab.n >= 1

    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower.genrou_jac_positions!(tab, J, dd, np)
    GradPower.genrou_jacobian_batch!(J, dp.zvec, dp.uvec, dp.pvec, tab, dd, np)

    J_fd = FiniteDiff.finite_difference_jacobian(dp.zvec) do z
        f = zeros(n)
        v = @view z[np+1:end]
        fv = @view f[np+1:end]
        mul!(fv, ps.network.ybus_real, v, -1.0, 0.0)
        GradPower.genrou_residual_batch!(f, z, dp.uvec, dp.pvec, tab, dd, np)
        return f
    end

    rows = Int[]
    for k in 1:tab.n
        dp_k = Int(tab.diff_ptr[k])
        append!(rows, dp_k:dp_k+5)
        ap_k = dd + Int(tab.alg_ptr[k])
        append!(rows, ap_k:ap_k+3)
    end
    @test _max_row_err(J, J_fd, rows) < 1e-5
end

@testset "FD Jacobian: IEESGO" begin
    ps, dp, L, dd, np, n = _setup("2bus.raw", "2bus_IEESGO.dyr")
    tab = L.ieesgo
    @test tab.n >= 1

    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower.ieesgo_jacobian_batch!(J, dp.pvec, tab, dd)

    J_fd = FiniteDiff.finite_difference_jacobian(dp.zvec) do z
        f = zeros(n)
        GradPower.ieesgo_residual_batch!(f, z, dp.pvec, tab, dd)
        return f
    end

    rows = Int[]
    for k in 1:tab.n
        dp_k = Int(tab.diff_ptr[k])
        append!(rows, dp_k:dp_k+4)
        ap_k = dd + Int(tab.alg_ptr[k])
        push!(rows, ap_k)
    end
    @test _max_row_err(J, J_fd, rows) < 1e-5
end

@testset "FD Jacobian: TGOV1" begin
    ps, dp, L, dd, np, n = _setup("2bus.raw", "2bus_TGOV1.dyr")
    tab = L.tgov1
    @test tab.n >= 1

    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower.tgov1_jacobian_batch!(J, dp.pvec, tab, dd)

    J_fd = FiniteDiff.finite_difference_jacobian(dp.zvec) do z
        f = zeros(n)
        GradPower.tgov1_residual_batch!(f, z, dp.pvec, tab, dd)
        return f
    end

    rows = Int[]
    for k in 1:tab.n
        dp_k = Int(tab.diff_ptr[k])
        append!(rows, dp_k:dp_k+1)
        ap_k = dd + Int(tab.alg_ptr[k])
        push!(rows, ap_k)
    end
    @test _max_row_err(J, J_fd, rows) < 1e-5
end

@testset "FD Jacobian: SEXS" begin
    ps, dp, L, dd, np, n = _setup("2bus.raw", "2bus_SEXS.dyr")
    tab = L.sexs
    @test tab.n >= 1

    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower.sexs_jacobian_batch!(J, dp.zvec, dp.pvec, tab)

    J_fd = FiniteDiff.finite_difference_jacobian(dp.zvec) do z
        f = zeros(n)
        GradPower.sexs_residual_batch!(f, z, dp.pvec, tab)
        return f
    end

    rows = Int[]
    for k in 1:tab.n
        dp_k = Int(tab.diff_ptr[k])
        append!(rows, dp_k:dp_k+1)
    end
    @test _max_row_err(J, J_fd, rows) < 1e-5
end

@testset "FD Jacobian: ESDC1A" begin
    ps, dp, L, dd, np, n = _setup("2bus.raw", "2bus_ESDC1A.dyr")
    tab = L.esdc1a
    @test tab.n >= 1

    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower.esdc1a_jacobian_batch!(J, dp.zvec, dp.pvec, tab)

    J_fd = FiniteDiff.finite_difference_jacobian(dp.zvec) do z
        f = zeros(n)
        GradPower.esdc1a_residual_batch!(f, z, dp.pvec, tab)
        return f
    end

    rows = Int[]
    for k in 1:tab.n
        dp_k = Int(tab.diff_ptr[k])
        append!(rows, dp_k:dp_k+2)
    end
    @test _max_row_err(J, J_fd, rows) < 1e-5
end

@testset "FD Jacobian: IEEEST" begin
    ps, dp, L, dd, np, n = _setup("2bus.raw", "2bus_IEEEST.dyr")
    tab = L.ieeest
    @test tab.n >= 1

    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower.ieeest_jacobian_batch!(J, dp.zvec, dp.pvec, tab, dd)

    J_fd = FiniteDiff.finite_difference_jacobian(dp.zvec) do z
        f = zeros(n)
        GradPower.ieeest_residual_batch!(f, z, dp.pvec, tab, dd)
        return f
    end

    rows = Int[]
    for k in 1:tab.n
        dp_k = Int(tab.diff_ptr[k])
        append!(rows, dp_k:dp_k+6)
        ap_k = dd + Int(tab.alg_ptr[k])
        push!(rows, ap_k)
    end
    @test _max_row_err(J, J_fd, rows) < 1e-5
end

@testset "FD Jacobian: ZIPLoad" begin
    ps, dp, L, dd, np, n = _setup("2bus.raw", "2bus.dyr")
    tab = L.zipload
    @test tab.n >= 1

    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower.zipload_jacobian_batch!(J, dp.zvec, dp.pvec, tab, np)

    J_fd = FiniteDiff.finite_difference_jacobian(dp.zvec) do z
        f = zeros(n)
        GradPower.zipload_residual_batch!(f, z, dp.pvec, tab, np)
        return f
    end

    # ZIPLoad writes only network rows (vr, vi per bus)
    rows = Int[]
    for k in 1:tab.n
        bus_k = Int(tab.bus[k])
        push!(rows, np + 2*(bus_k-1) + 1)
        push!(rows, np + 2*(bus_k-1) + 2)
    end
    @test _max_row_err(J, J_fd, rows) < 1e-5
end

# StaticGen FD test skipped: ACTIVSg2000 (9694 unknowns) makes FD
# Jacobian prohibitively slow. Covered by test_device_kernels.jl
# acceptance tests (residual ≈ 0, sparsity check) instead.

# -----------------------------------------------------------------------
# Full-system FD Jacobian (catches cross-device coupling bugs)
# -----------------------------------------------------------------------

@testset "FD Jacobian: full system (2bus)" begin
    ps, dp, L, dd, np, n = _setup("2bus.raw", "2bus.dyr")
    dyn = ps.dynamic

    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower._rhs_jac_batched!(J, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    J_fd = FiniteDiff.finite_difference_jacobian(dp.zvec) do z
        f = zeros(n)
        u = copy(dp.uvec)
        GradPower._rhs_fun_batched!(f, z, u, dp.pvec, dyn, ps.network.ybus_real, L)
        return f
    end

    @test maximum(abs, Array(J) - J_fd) < 1e-5
end

@testset "FD Jacobian: full system (ieee9 with governors)" begin
    ps, dp, L, dd, np, n = _setup("ieee9_v33.raw", "ieee9bus_gov.dyr")
    dyn = ps.dynamic

    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower._rhs_jac_batched!(J, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    J_fd = FiniteDiff.finite_difference_jacobian(dp.zvec) do z
        f = zeros(n)
        u = copy(dp.uvec)
        GradPower._rhs_fun_batched!(f, z, u, dp.pvec, dyn, ps.network.ybus_real, L)
        return f
    end

    @test maximum(abs, Array(J) - J_fd) < 1e-5
end
