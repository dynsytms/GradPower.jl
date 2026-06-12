# Phase 2.1 (ROADMAP §3 Phase 2.1) — A2.0 parity test.
#
# Asserts that the new batched GENROU kernels (`genrou_residual_batch!`,
# `genrou_jacobian_batch!`, `genrou_jac_positions!`) produce numerically
# identical output to the legacy heterogeneous loop in
# `src/dynamics.jl::rhs_fun!` / `rhs_jac!` for GENROU contributions on
# the existing validation cases.
#
# This test is the design-of-A2.0 acceptance gate. Other Phase 2.1
# kernels (IEESGO, ESDC1A, ZIPLoad, network) will land with their own
# parity tests cloned from this one.

@testset "Phase 2.1 A2.0: GENROU residual batched kernel parity" begin
    cases = [
        ("ieee9 no governor", "examples/ieee9_v33.raw", "examples/ieee9bus.dyr"),
        ("ieee9 with governor", "examples/ieee9_v33.raw", "examples/ieee9bus_gov.dyr"),
        ("2bus", "examples/2bus.raw", "examples/2bus.dyr"),
    ]

    for (label, raw, dyr) in cases
        @testset "$label" begin
            ps = from_psse(joinpath(@__DIR__, "..", raw),
                           joinpath(@__DIR__, "..", dyr))
            GradPower.build_network!(ps)
            GradPower.runpf(ps)
            dp = GradPower.DynamicProblem(ps)
            GradPower.initialize_dynamics!(dp, ps)

            diff_dim = ps.dynamic.diff_dim
            alg_dim  = ps.dynamic.alg_dim
            net_ptr  = diff_dim + alg_dim
            nbus     = length(ps.buses)
            sysdim   = diff_dim + alg_dim + 2*nbus

            # --- f_old via the legacy heterogeneous loop -------------
            f_old = zeros(Float64, sysdim)
            GradPower.rhs_fun!(f_old, dp.zvec, dp.uvec, dp.pvec, ps)

            # --- f_new: replay just the GENROU contribution on a fresh f
            # Initialize with the network -ybus*v (matches what the
            # legacy `rhs_fun!` does to the voltage block) and then run
            # only the GENROU batch on top. Compare GENROU-touched rows
            # against `f_old`.
            f_new = zeros(Float64, sysdim)
            v = @view dp.zvec[net_ptr+1:end]
            fv = @view f_new[net_ptr+1:end]
            mul!(fv, ps.network.ybus_real, v, -1.0, 0.0)
            GradPower.genrou_residual_batch!(
                f_new, dp.zvec, dp.uvec, dp.pvec,
                ps.dynamic.layout.genrou, diff_dim, net_ptr,
            )

            # On rows owned by GENROU (diff + alg) the two paths must
            # agree exactly. Network rows in f_new only carry GENROU
            # injections (no ZIPLoad), so they will NOT match f_old
            # unless we also account for loads — we only check GENROU
            # rows here; the full-system parity gate comes once IEESGO,
            # ZIPLoad, and network kernels also exist.
            tab = ps.dynamic.layout.genrou
            max_delta_diff = 0.0
            max_delta_alg  = 0.0
            for k in 1:tab.n
                dp_k = Int(tab.diff_ptr[k])
                ap_k = Int(tab.alg_ptr[k])
                for j in 0:5
                    max_delta_diff = max(max_delta_diff,
                                         abs(f_old[dp_k+j] - f_new[dp_k+j]))
                end
                for j in 0:3
                    row = diff_dim + ap_k + j
                    max_delta_alg = max(max_delta_alg,
                                        abs(f_old[row] - f_new[row]))
                end
            end
            @test max_delta_diff < 1e-14
            @test max_delta_alg  < 1e-14
        end
    end
end

@testset "Phase 2.1 A2.0: GENROU Jacobian batched kernel parity" begin
    cases = [
        ("ieee9 no governor", "examples/ieee9_v33.raw", "examples/ieee9bus.dyr"),
        ("ieee9 with governor", "examples/ieee9_v33.raw", "examples/ieee9bus_gov.dyr"),
        ("2bus", "examples/2bus.raw", "examples/2bus.dyr"),
    ]

    for (label, raw, dyr) in cases
        @testset "$label" begin
            ps = from_psse(joinpath(@__DIR__, "..", raw),
                           joinpath(@__DIR__, "..", dyr))
            GradPower.build_network!(ps)
            GradPower.runpf(ps)
            dp = GradPower.DynamicProblem(ps)
            GradPower.initialize_dynamics!(dp, ps)

            diff_dim = ps.dynamic.diff_dim
            alg_dim  = ps.dynamic.alg_dim
            net_ptr  = diff_dim + alg_dim

            # --- J_old via legacy ------------------------------------
            J_old = GradPower.preallocate_jacobian(ps)
            fill!(J_old.nzval, 0.0)
            GradPower.rhs_jac!(J_old, dp.zvec, dp.uvec, dp.pvec, ps)

            # --- J_new: same sparsity, then GENROU batch ------------
            J_new = GradPower.preallocate_jacobian(ps)
            fill!(J_new.nzval, 0.0)
            GradPower.genrou_jac_positions!(
                ps.dynamic.layout.genrou, J_new, diff_dim, net_ptr,
            )
            GradPower.genrou_jacobian_batch!(
                J_new, dp.zvec, dp.uvec, dp.pvec,
                ps.dynamic.layout.genrou, diff_dim, net_ptr,
            )

            # Compare GENROU-owned ROWS exactly. Diff rows + alg rows
            # are 1:1 with one device (no other writer), so they must
            # match J_old exactly on those rows. Network rows in J_new
            # only carry GENROU injection; check only the entries
            # GENROU is responsible for via jac_pos[k, J_GR_NR_*].
            tab = ps.dynamic.layout.genrou
            max_delta = 0.0
            for k in 1:tab.n
                dp_k = Int(tab.diff_ptr[k])
                ap_k = Int(tab.alg_ptr[k])
                # Diff rows + alg rows
                for col in 1:size(J_old, 2)
                    for row in [dp_k:dp_k+5;
                                diff_dim + ap_k : diff_dim + ap_k + 3]
                        max_delta = max(max_delta,
                                        abs(J_old[row, col] - J_new[row, col]))
                    end
                end
                # GENROU's six network entries (vr-row delta/iq/id and
                # vi-row delta/iq/id) — read them by jac_pos and check
                # against J_old at the same global position.
                for slot in (GradPower.J_GR_NR_delta, GradPower.J_GR_NR_iq, GradPower.J_GR_NR_id,
                             GradPower.J_GR_NI_delta, GradPower.J_GR_NI_iq, GradPower.J_GR_NI_id)
                    pos = tab.jac_pos[k, slot]
                    @assert pos != 0
                    max_delta = max(max_delta,
                                    abs(J_new.nzval[pos] - J_old.nzval[pos]))
                end
            end
            @test max_delta < 1e-14
        end
    end
end
