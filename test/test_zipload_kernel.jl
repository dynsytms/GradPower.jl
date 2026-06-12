# Phase 2.1: ZIPLoad batched kernel parity test.
#
# ZIPLoad has full legacy code (cinject! + rhs_jac!) so this gets a
# strict parity gate vs the heterogeneous loop: max|Δ| < 1e-14 for both
# residual (network rows) and Jacobian (the 4 entries per load).

@testset "Phase 2.1: ZIPLoad residual batched kernel parity" begin
    cases = [
        ("ieee9 no gov", "examples/ieee9_v33.raw", "examples/ieee9bus.dyr"),
        ("ieee9 gov",    "examples/ieee9_v33.raw", "examples/ieee9bus_gov.dyr"),
        ("2bus",         "examples/2bus.raw",      "examples/2bus.dyr"),
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

            # f_old via legacy
            f_old = zeros(sysdim)
            GradPower.rhs_fun!(f_old, dp.zvec, dp.uvec, dp.pvec, ps)

            # f_new: zero, run ONLY the ZIPLoad batch on top
            f_new = zeros(sysdim)
            GradPower.zipload_residual_batch!(
                f_new, dp.zvec, dp.pvec, ps.dynamic.layout.zipload, net_ptr,
            )

            # f_old's network block contains: -ybus*v + all device cinject!
            # accumulations (including ZIPLoad). f_new's network block
            # contains ONLY the ZIPLoad accumulation. Subtract: residual
            # at each load's bus must match the per-bus contribution.
            # Simpler check: compare a freshly-built f_new+f_legacy_other
            # against f_old, or just check that the load buses are
            # touched correctly.
            #
            # Instead use the cleanest approach: rebuild f_old via the
            # legacy path with loads only by zeroing other devices...
            # too complex. Use FD check: at any bus with a load, f_new's
            # contribution at that bus should equal f_old - (legacy
            # without ZIPLoad). Better: write a focused test using a
            # mock z where we know what the result should be.
            #
            # Pragmatic check: extract the ZIPLoad contribution from the
            # legacy by running cinject! manually and compare to f_new.
            f_ref = zeros(sysdim)
            for (i, device) in enumerate(ps.dynamic.devices)
                if device.dtype isa GradPower.ZIPLoad
                    bus = ps.dynamic.map.bus[i]
                    par_ptr = ps.dynamic.map.par_ptr[i]
                    par_size = ps.dynamic.map.par_size[i]
                    par = @view dp.pvec[par_ptr:par_ptr+par_size-1]
                    vloc = @view dp.zvec[net_ptr + 2*(bus-1) + 1 : net_ptr + 2*(bus-1) + 2]
                    f_net = @view f_ref[net_ptr + 2*(bus-1) + 1 : net_ptr + 2*(bus-1) + 2]
                    GradPower.cinject!(f_net, dp.zvec, dp.zvec, dp.uvec, par, vloc, device.dtype)
                end
            end

            max_delta = maximum(abs, f_new .- f_ref)
            @test max_delta < 1e-14
        end
    end
end

@testset "Phase 2.2: ZIPLoad Jacobian — self-consistency vs FD" begin
    # Post Phase 2.2 there is no legacy per-device ZIPLoad Jacobian
    # method to compare against. Instead validate the batched kernel
    # by finite-differencing the residual at each load's bus and
    # asserting J*Δv ≈ Δf for small perturbations.
    cases = [
        ("ieee9 no gov", "examples/ieee9_v33.raw", "examples/ieee9bus.dyr"),
        ("2bus",         "examples/2bus.raw",      "examples/2bus.dyr"),
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
            n = length(dp.zvec)

            # Build the analytic ZIPLoad Jacobian
            J_new = GradPower.preallocate_jacobian(ps)
            fill!(J_new.nzval, 0.0)
            GradPower.zipload_jacobian_batch!(
                J_new, dp.zvec, dp.pvec, ps.dynamic.layout.zipload, net_ptr,
            )

            # Build the FD ZIPLoad Jacobian by perturbing each voltage
            # entry in turn and recording the change in the residual
            # written by zipload_residual_batch!.
            ε = 1e-7
            f0 = zeros(n)
            GradPower.zipload_residual_batch!(f0, dp.zvec, dp.pvec, ps.dynamic.layout.zipload, net_ptr)

            tab = ps.dynamic.layout.zipload
            max_err = 0.0
            for k in 1:tab.n
                bus = Int(tab.bus[k])
                vr_idx = net_ptr + 2*(bus - 1) + 1
                vi_idx = vr_idx + 1

                # FD column for vr
                z_p = copy(dp.zvec); z_p[vr_idx] += ε
                z_m = copy(dp.zvec); z_m[vr_idx] -= ε
                f_p = zeros(n); GradPower.zipload_residual_batch!(f_p, z_p, dp.pvec, tab, net_ptr)
                f_m = zeros(n); GradPower.zipload_residual_batch!(f_m, z_m, dp.pvec, tab, net_ptr)
                fd_vr_vr = (f_p[vr_idx] - f_m[vr_idx]) / (2ε)
                fd_vi_vr = (f_p[vi_idx] - f_m[vi_idx]) / (2ε)

                # FD column for vi
                z_p = copy(dp.zvec); z_p[vi_idx] += ε
                z_m = copy(dp.zvec); z_m[vi_idx] -= ε
                f_p = zeros(n); GradPower.zipload_residual_batch!(f_p, z_p, dp.pvec, tab, net_ptr)
                f_m = zeros(n); GradPower.zipload_residual_batch!(f_m, z_m, dp.pvec, tab, net_ptr)
                fd_vr_vi = (f_p[vr_idx] - f_m[vr_idx]) / (2ε)
                fd_vi_vi = (f_p[vi_idx] - f_m[vi_idx]) / (2ε)

                an_vr_vr = J_new.nzval[tab.jac_pos[k, GradPower.J_ZL_VR_VR]]
                an_vr_vi = J_new.nzval[tab.jac_pos[k, GradPower.J_ZL_VR_VI]]
                an_vi_vr = J_new.nzval[tab.jac_pos[k, GradPower.J_ZL_VI_VR]]
                an_vi_vi = J_new.nzval[tab.jac_pos[k, GradPower.J_ZL_VI_VI]]

                max_err = max(max_err,
                              abs(fd_vr_vr - an_vr_vr),
                              abs(fd_vr_vi - an_vr_vi),
                              abs(fd_vi_vr - an_vi_vr),
                              abs(fd_vi_vi - an_vi_vi))
            end
            @test max_err < 1e-6
        end
    end
end
