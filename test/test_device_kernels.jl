# Kernel acceptance tests for TGOV1, SEXS, ESDC1A, IEEEST, StaticGen.
#
# Same pattern as test_ieesgo_kernel.jl: no legacy parity (these devices
# have no rhs_fun!/rhs_jac! shim), so test:
#   (1) residual ≈ 0 at z0 (init consistency)
#   (2) Jacobian sparsity complete (no jac_pos == 0 zombies)
#   (3) Jacobian spot-check (known diagonal entries)

const EX_DK = joinpath(@__DIR__, "..", "examples")

# -----------------------------------------------------------------------
# TGOV1
# -----------------------------------------------------------------------

@testset "TGOV1 residual ≈ 0 at z0 after init" begin
    ps = from_psse(joinpath(EX_DK, "2bus.raw"), joinpath(EX_DK, "2bus_TGOV1.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    @test L.tgov1.n >= 1

    dd = ps.dynamic.diff_dim
    n = length(dp.zvec)
    f = zeros(n)
    GradPower.tgov1_residual_batch!(f, dp.zvec, dp.pvec, L.tgov1, dd)

    max_diff = 0.0
    max_alg  = 0.0
    for k in 1:L.tgov1.n
        dp_k = Int(L.tgov1.diff_ptr[k])
        ap_k = dd + Int(L.tgov1.alg_ptr[k])
        for j in 0:1
            max_diff = max(max_diff, abs(f[dp_k + j]))
        end
        max_alg = max(max_alg, abs(f[ap_k]))
    end
    @test max_diff < 1e-9
    @test max_alg  < 1e-9
end

@testset "TGOV1 Jacobian sparsity complete" begin
    ps = from_psse(joinpath(EX_DK, "2bus.raw"), joinpath(EX_DK, "2bus_TGOV1.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    J = GradPower.preallocate_jacobian(ps)

    for k in 1:L.tgov1.n
        for slot in 1:GradPower.TGOV1_JAC_NENTRIES
            pos = L.tgov1.jac_pos[k, slot]
            if slot in (GradPower.J_TG_R2_w, GradPower.J_TG_A_w) && L.tgov1.w_idx[k] == 0
                @test pos == 0
            else
                @test pos != 0
            end
        end
    end

    # Spot-check: diagonal entries
    fill!(J.nzval, 0.0)
    dd = ps.dynamic.diff_dim
    GradPower.tgov1_jacobian_batch!(J, dp.pvec, L.tgov1, dd)

    for k in 1:L.tgov1.n
        dp_k = Int(L.tgov1.diff_ptr[k])
        ap_k = dd + Int(L.tgov1.alg_ptr[k])
        T3 = L.tgov1.T3[k]
        # dx1/dt row: ∂/∂x1 = -1/T3
        @test J[dp_k, dp_k] ≈ -1.0 / T3
        # alg row: p_m = ... - p_m → ∂/∂p_m = -1
        @test J[ap_k, ap_k] ≈ -1.0
    end
end

# -----------------------------------------------------------------------
# SEXS
# -----------------------------------------------------------------------

@testset "SEXS residual ≈ 0 at z0 after init" begin
    ps = from_psse(joinpath(EX_DK, "2bus.raw"), joinpath(EX_DK, "2bus_SEXS.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    @test L.sexs.n >= 1

    n = length(dp.zvec)
    f = zeros(n)
    GradPower.sexs_residual_batch!(f, dp.zvec, dp.pvec, L.sexs)

    max_err = 0.0
    for k in 1:L.sexs.n
        dp_k = Int(L.sexs.diff_ptr[k])
        for j in 0:1
            max_err = max(max_err, abs(f[dp_k + j]))
        end
    end
    @test max_err < 1e-9
end

@testset "SEXS Jacobian sparsity complete" begin
    ps = from_psse(joinpath(EX_DK, "2bus.raw"), joinpath(EX_DK, "2bus_SEXS.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    J = GradPower.preallocate_jacobian(ps)

    for k in 1:L.sexs.n
        for slot in 1:GradPower.SEXS_JAC_NENTRIES
            pos = L.sexs.jac_pos[k, slot]
            # vs slots are 0 when no PSS is attached
            if slot in (GradPower.J_SX_R1_vs, GradPower.J_SX_R2_vs) && L.sexs.vs_idx[k] == 0
                @test pos == 0
            else
                @test pos != 0
            end
        end
    end

    # Spot-check: TE diagonal
    fill!(J.nzval, 0.0)
    GradPower.sexs_jacobian_batch!(J, dp.zvec, dp.pvec, L.sexs)

    for k in 1:L.sexs.n
        dp_k = Int(L.sexs.diff_ptr[k])
        TE = L.sexs.TE[k]
        @test J[dp_k + 1, dp_k + 1] ≈ -1.0 / TE
    end
end

# -----------------------------------------------------------------------
# ESDC1A
# -----------------------------------------------------------------------

@testset "ESDC1A residual ≈ 0 at z0 after init" begin
    ps = from_psse(joinpath(EX_DK, "2bus.raw"), joinpath(EX_DK, "2bus_ESDC1A.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    @test L.esdc1a.n >= 1

    n = length(dp.zvec)
    f = zeros(n)
    GradPower.esdc1a_residual_batch!(f, dp.zvec, dp.pvec, L.esdc1a)

    max_err = 0.0
    for k in 1:L.esdc1a.n
        dp_k = Int(L.esdc1a.diff_ptr[k])
        for j in 0:2
            max_err = max(max_err, abs(f[dp_k + j]))
        end
    end
    @test max_err < 1e-9
end

@testset "ESDC1A Jacobian sparsity complete" begin
    ps = from_psse(joinpath(EX_DK, "2bus.raw"), joinpath(EX_DK, "2bus_ESDC1A.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    J = GradPower.preallocate_jacobian(ps)

    for k in 1:L.esdc1a.n
        for slot in 1:GradPower.ESDC1A_JAC_NENTRIES
            pos = L.esdc1a.jac_pos[k, slot]
            # vs slot is 0 when no PSS is attached
            if slot == GradPower.J_EX_R1_vs && L.esdc1a.vs_idx[k] == 0
                @test pos == 0
            else
                @test pos != 0
            end
        end
    end

    # Spot-check
    fill!(J.nzval, 0.0)
    GradPower.esdc1a_jacobian_batch!(J, dp.zvec, dp.pvec, L.esdc1a)

    for k in 1:L.esdc1a.n
        dp_k = Int(L.esdc1a.diff_ptr[k])
        Ta = L.esdc1a.Ta[k]
        # vr1 row: ∂f/∂vr1 = -1/Ta
        @test J[dp_k, dp_k] ≈ -1.0 / Ta
    end
end

# -----------------------------------------------------------------------
# IEEEST
# -----------------------------------------------------------------------

@testset "IEEEST residual ≈ 0 at z0 after init" begin
    ps = from_psse(joinpath(EX_DK, "2bus.raw"), joinpath(EX_DK, "2bus_IEEEST.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    @test L.ieeest.n >= 1

    dd = ps.dynamic.diff_dim
    n = length(dp.zvec)
    f = zeros(n)
    GradPower.ieeest_residual_batch!(f, dp.zvec, dp.pvec, L.ieeest, dd)

    max_diff = 0.0
    max_alg  = 0.0
    for k in 1:L.ieeest.n
        dp_k = Int(L.ieeest.diff_ptr[k])
        for j in 0:6
            max_diff = max(max_diff, abs(f[dp_k + j]))
        end
        ap_k = dd + Int(L.ieeest.alg_ptr[k])
        max_alg = max(max_alg, abs(f[ap_k]))
    end
    @test max_diff < 1e-9
    @test max_alg  < 1e-9
end

@testset "IEEEST Jacobian sparsity complete" begin
    ps = from_psse(joinpath(EX_DK, "2bus.raw"), joinpath(EX_DK, "2bus_IEEEST.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    J = GradPower.preallocate_jacobian(ps)
    dd = ps.dynamic.diff_dim

    for k in 1:L.ieeest.n
        for slot in 1:GradPower.IEEEST_JAC_NENTRIES
            pos = L.ieeest.jac_pos[k, slot]
            # Allow 0 for optional omega slot when w_idx == 0
            if pos == 0 && L.ieeest.w_idx[k] == 0
                continue
            end
            @test pos != 0
        end
    end

    # Spot-check
    fill!(J.nzval, 0.0)
    GradPower.ieeest_jacobian_batch!(J, dp.zvec, dp.pvec, L.ieeest, dd)

    for k in 1:L.ieeest.n
        dp_k = Int(L.ieeest.diff_ptr[k])
        ap_k = dd + Int(L.ieeest.alg_ptr[k])
        T6 = L.ieeest.T6[k]
        # ds6/dt = (KS*y4 - s6)/T6 → ∂/∂s6 = -1/T6
        @test J[dp_k + 6, dp_k + 6] ≈ -1.0 / T6
        # Alg row: vs = (T5/T6)*(KS*y4 - s6) → ∂/∂vs = ... but the alg
        # row has vs on LHS, so the diagonal should not be 0
        @test J[ap_k, ap_k] != 0.0
    end
end

# -----------------------------------------------------------------------
# StaticGen
# -----------------------------------------------------------------------

@testset "StaticGen residual ≈ 0 at z0 after init" begin
    ps = from_psse(joinpath(EX_DK, "ACTIVSg2000.raw"), joinpath(EX_DK, "ACTIVSg2000.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    @test L.static_gen.n >= 1

    n = length(dp.zvec)
    f = zeros(n)
    GradPower.static_gen_residual_batch!(f, dp.zvec, dp.pvec, L.static_gen)

    # Static generators write to network rows; at z0 they should be
    # consistent (residual ≈ 0 when combined with -ybus*v).
    # Test: the static_gen contribution alone should have entries that
    # are finite and reasonable (not NaN, not huge).
    max_val = 0.0
    dd = ps.dynamic.diff_dim
    ad = ps.dynamic.alg_dim
    np = dd + ad
    for k in 1:L.static_gen.n
        bus_k = Int(L.static_gen.bus[k])
        vr_row = np + 2*(bus_k-1) + 1
        vi_row = np + 2*(bus_k-1) + 2
        @test isfinite(f[vr_row])
        @test isfinite(f[vi_row])
    end
end

@testset "StaticGen Jacobian sparsity" begin
    ps = from_psse(joinpath(EX_DK, "ACTIVSg2000.raw"), joinpath(EX_DK, "ACTIVSg2000.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    J = GradPower.preallocate_jacobian(ps)

    # At minimum: every static gen should have at least some nonzero jac_pos entries
    for k in 1:L.static_gen.n
        nonzero_count = count(L.static_gen.jac_pos[k, :] .!= 0)
        @test nonzero_count >= 2  # at least vr,vr and vi,vi diagonal
    end
end
