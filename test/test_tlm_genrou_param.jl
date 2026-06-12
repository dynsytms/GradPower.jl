# Phase 2.1 acceptance gate: TLM gradient w.r.t. GENROU.H on the 2-bus
# governor case (GENROU + IEESGO) matches finite-difference.
#
# Why this test: this is the smallest case that exercises the Phase 2.1
# batched-kernel dispatcher AND a non-trivial control wiring (governor
# producing p_m). If the batched Jacobian regressed (wrong jac_pos
# write, wrong parameter slot, dropped contribution), TLM and FD would
# diverge. We test w.r.t. a GENROU parameter (not an IEESGO parameter)
# because the legacy `jacp_vec!` AD path only knows about devices with
# `rhs_alg!`/`rhs_diff!`/`cinject!` methods — IEESGO doesn't have those
# (pre-existing gap that predates Phase 2.1; see ROADMAP "Known
# Issues"). Pushing parameter sensitivity through the batched kernels
# directly is a separate Phase 4 task.

@testset "Phase 2.1 gate: TLM ≈ FD for GENROU.H on 2-bus governor" begin
    raw_file = joinpath(@__DIR__, "..", "examples", "2bus.raw")
    dyr_file = joinpath(@__DIR__, "..", "examples", "2bus_IEESGO.dyr")

    function build()
        ps = GradPower.from_psse(raw_file, dyr_file)
        GradPower.build_network!(ps)
        GradPower.runpf(ps)
        dp = GradPower.DynamicProblem(ps)
        GradPower.initialize_dynamics!(dp, ps)
        return ps, dp
    end

    # baseline trajectory
    ps, dp = build()
    GradPower.add_event!(ps, GradPower.ContingencyEvent(2, 0.02, 0.2, 0.3))
    tfinal = 1.0
    tvec, traj = GradPower.integrate!(dp, ps, tfinal)

    # Locate GENROU device and its H slot in pvec.
    # Genrou.fill_pvec! layout (see src/generators.jl): pvec offsets
    #   0:x_d 1:x_q 2:x_dp 3:x_qp 4:x_ddp 5:xl 6:H 7:D ...
    gen_idx = findfirst(d -> d.dtype isa GradPower.Genrou, ps.dynamic.devices)
    @assert gen_idx !== nothing "no GENROU device"
    par_ptr = ps.dynamic.map.par_ptr[gen_idx]
    h_slot  = par_ptr + 6

    # TLM: δp = e_H
    δp = zeros(length(dp.pvec))
    δp[h_slot] = 1.0
    δz0 = zeros(length(dp.zvec))
    δztf_tlm = GradPower.tlm(δz0, dp, ps, traj, tvec, δp=δp)

    # FD: perturb pvec[h_slot], re-integrate, central difference
    function final_state(ε)
        ps2, dp2 = build()
        GradPower.add_event!(ps2, GradPower.ContingencyEvent(2, 0.02, 0.2, 0.3))
        dp2.pvec[h_slot] += ε
        _, traj_e = GradPower.integrate!(dp2, ps2, tfinal)
        return traj_e[:, end]
    end

    ε = 1e-3
    zp = final_state( ε)
    zm = final_state(-ε)
    δztf_fd = (zp .- zm) ./ (2ε)

    @test isapprox(δztf_tlm, δztf_fd, rtol=1e-3)
end
