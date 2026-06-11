# Regression test for the adjoint event-offset alignment fix.
#
# The forward integrator (`src/dynamics.jl::integrate!`) activates the fault
# event at `k == step_on` (after Newton at step `k=step_on` has completed and
# `traj[:, step_on+1]` is stored) and deactivates at `k == step_off`.
#
# `traj[:, i]` corresponds to the state at the END of forward step `i-1`.
# Therefore the Jacobian built at `traj[:, i]` in the adjoint loop matches the
# fault state DURING forward step `k = i-1`. The fault is ACTIVE during
# forward steps `k ∈ [step_on+1, step_off]`, so the adjoint Jacobian at
# `i ∈ [step_on+2, step_off+1]` must be evaluated with the fault active.
#
# The adjoint loop applies activate/deactivate at the END of the iteration
# (after the Jacobian for that iteration has been built and consumed). Hence
# to make iteration `i = step_off+1` see fault ACTIVE, we activate at end of
# `i = step_off+2`; to make iteration `i = step_on+1` see fault INACTIVE, we
# deactivate at end of `i = step_on+2`.
#
# This test compares the adjoint gradient of sum(ω²) (the built-in
# `functional`) w.r.t. the generator's H parameter against a centered
# finite-difference gradient. With the correct `+2` offsets, the relative
# error is < 1e-6. With `+1` or `+0` offsets the error blows up to O(1e-1)
# (verified empirically: changing both `+2` -> `+1` in src/sensitivities.jl
# pushes λ relative error from ~1.5e-9 to ~0.15).

@testset "adjoint event offset" begin
    raw_file = "testdata/2bus.raw"
    dyr_file = "testdata/2bus.dyr"

    raw = GradPower.read_psse_raw(raw_file)
    sys = GradPower.raw_to_grad(raw)
    psd = GradPower.PowerSystemDynamics(dyr_file)
    GradPower.set_dynamics!(sys, psd)
    GradPower.build_network!(sys)
    GradPower.runpf!(sys, verbose=false)

    tfinal = 4.8
    dt = 1.0/120.0

    # Locate target parameter: Genrou.H of device 1 (parameter slot 7).
    @assert psd.devices[1].dtype isa GradPower.Genrou
    par_idx = psd.devices[1].par_ptr + 6  # H is the 7th Genrou parameter

    # Functional gradient via adjoint.
    function adjoint_grad()
        dprob = GradPower.DynamicProblem(sys)
        GradPower.initialize_dynamics!(dprob, sys)
        # Re-register the event each call: add_event! mutates psd.events.
        empty!(psd.events)
        GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.02, 0.2, 0.3))
        tvec, traj = GradPower.integrate!(dprob, sys, tfinal, dt=dt)
        λ0 = zeros(length(dprob.zvec))
        _, μ, _ = GradPower.adjoint(λ0, dprob, sys, traj, tvec,
                                    functional=true, store_trajectory=true)
        return μ[par_idx]
    end

    # Functional value (used for FD).
    function functional_value(par_val)
        dprob = GradPower.DynamicProblem(sys)
        GradPower.initialize_dynamics!(dprob, sys)
        dprob.pvec[par_idx] = par_val
        empty!(psd.events)
        GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.02, 0.2, 0.3))
        tvec, traj = GradPower.integrate!(dprob, sys, tfinal, dt=dt)
        val = 0.0
        for j in 1:(size(traj, 2) - 1)
            rfun = GradPower.functional(traj[:, j + 1], dprob.uvec, dprob.pvec, sys)
            val += (tvec[j + 1] - tvec[j]) * rfun
        end
        return val
    end

    grad_adj = adjoint_grad()

    # Centered FD reference around the nominal value of H.
    dprob_ref = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob_ref, sys)
    p_nom = dprob_ref.pvec[par_idx]
    h = 1e-5 * max(abs(p_nom), 1.0)
    grad_fd = (functional_value(p_nom + h) - functional_value(p_nom - h)) / (2*h)

    rel_err = abs(grad_adj - grad_fd) / max(abs(grad_fd), 1e-12)
    @test rel_err < 1e-6
end
