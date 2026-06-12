# IEESGO kernel acceptance test.
#
# The legacy `rhs_fun!`/`rhs_jac!`/`cinject!` methods don't exist for
# IEESGO — the heterogeneous loop hits `@warn` fallbacks and IEESGO's
# residual contributions silently drop. So there's no Julia-side parity
# to assert. The acceptance gate is:
#   (1) After init, IEESGO residual rows are ≈ 0 at z0 (consistency
#       between the kernel and `initialize_dynamics!`).
#   (2) The Jacobian sparsity pattern includes every (row, col) entry
#       the kernel writes (no `jac_pos == 0` zombies).
#   (3) Built into J via the new `ieesgo_preallocate!` hook.

@testset "IEESGO residual ≈ 0 at z0 after init" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout
    @test L.ieesgo.n == 3

    # Reset f, run JUST the IEESGO batched residual on top.
    n = length(dp.zvec)
    f = zeros(n)
    diff_dim = ps.dynamic.diff_dim
    GradPower.ieesgo_residual_batch!(f, dp.zvec, dp.pvec, L.ieesgo, diff_dim)

    # Every governor's 5 diff rows + 1 alg row must be ≈ 0 (steady state).
    max_diff = 0.0
    max_alg  = 0.0
    for k in 1:L.ieesgo.n
        dp_k = Int(L.ieesgo.diff_ptr[k])
        ap_k = Int(L.ieesgo.alg_ptr[k]) + diff_dim
        for j in 0:4
            max_diff = max(max_diff, abs(f[dp_k + j]))
        end
        max_alg = max(max_alg, abs(f[ap_k]))
    end
    @test max_diff < 1e-9
    @test max_alg  < 1e-9
end

@testset "IEESGO Jacobian sparsity is complete" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    L = ps.dynamic.layout

    # preallocate_jacobian both builds J's sparsity AND fills jac_pos.
    J = GradPower.preallocate_jacobian(ps)

    # Every IEESGO slot must be populated (nonzero — 0 is the
    # "sparsity-missing" sentinel except for the w_idx slot when the
    # governor has no wired generator).
    for k in 1:L.ieesgo.n
        for slot in 1:GradPower.IEESGO_JAC_NENTRIES
            pos = L.ieesgo.jac_pos[k, slot]
            # The w slot may legitimately be 0 if w_idx == 0 (orphan).
            if slot == GradPower.J_IG_R1_w && L.ieesgo.w_idx[k] == 0
                @test pos == 0
            else
                @test pos != 0
            end
        end
    end

    # Fill Jacobian, then assert all written entries land at the claimed
    # (row, col) by reading back J[row, col].
    fill!(J.nzval, 0.0)
    GradPower.ieesgo_jacobian_batch!(J, dp.pvec, L.ieesgo, ps.dynamic.diff_dim)
    diff_dim = ps.dynamic.diff_dim

    for k in 1:L.ieesgo.n
        dp_k = Int(L.ieesgo.diff_ptr[k])
        ap_k = Int(L.ieesgo.alg_ptr[k]) + diff_dim
        T1 = L.ieesgo.T1[k]; T3 = L.ieesgo.T3[k]; T4 = L.ieesgo.T4[k]
        T5 = L.ieesgo.T5[k]; T6 = L.ieesgo.T6[k]
        K2 = L.ieesgo.K2[k]; K3 = L.ieesgo.K3[k]

        @test J[dp_k, dp_k]         ≈ -1.0 / T1
        @test J[dp_k + 1, dp_k]     ≈ (1.0 - L.ieesgo.T2[k] / T3) / T3
        @test J[dp_k + 1, dp_k + 1] ≈ -1.0 / T3
        @test J[dp_k + 2, dp_k + 2] ≈ -1.0 / T4
        @test J[dp_k + 3, dp_k + 2] ≈ K2 / T5
        @test J[dp_k + 4, dp_k + 3] ≈ K3 / T6
        @test J[ap_k, dp_k + 2]     ≈ 1.0 - K2
        @test J[ap_k, ap_k]         ≈ -1.0
    end
end
