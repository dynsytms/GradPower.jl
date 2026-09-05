#!/usr/bin/env julia
#
# ===========================================================================
#  TUTORIAL: generating transient-stability data with GradPower.jl
# ===========================================================================
#
# Read this file top to bottom. It runs a small sweep in a single process --
# no MPI, no HDF5 -- so you can see every step of what the production driver
# (scripts/generate_dynamics.jl) does at scale.
#
#   julia --project=scripts scripts/tutorial_generate_data.jl
#
# One data point ("scenario") is one time-domain simulation, defined by three
# inputs:
#
#   1. LOADING          lambda, a scalar multiplying every load in the system.
#                       Sampled from a distribution -- this is what makes the
#                       dataset a Monte Carlo sample of operating points
#                       rather than a fixed grid.
#   2. FAULT LOCATION   which bus the fault is applied to.
#   3. FAULT ADMITTANCE r_fault, a shunt resistance to ground. The solver uses
#                       y = 1/r_fault, so SMALLER r means MORE admittance.
#
# The output is a trajectory plus a stability label.
#
# ---------------------------------------------------------------------------
#  Read this before you trust a single number
# ---------------------------------------------------------------------------
#
# * r_fault severity is NOT monotonic. The fault is purely RESISTIVE, so the
#   real power it absorbs is V^2/r_fault. As r_fault -> 0 the voltage collapses
#   but the absorbed power -> 0; as r_fault -> infinity nothing happens. Peak
#   disturbance is in between, where the fault resistance roughly matches the
#   network's Thevenin impedance at that bus. On IEEE9 at bus 7 the worst
#   r_fault is near 0.1, NOT the smallest value. Sweep r_fault on a log grid
#   and check where severity actually peaks for YOUR case before committing to
#   a range.
#
# * Unstable trajectories are chaotic. Refining dt does not make an unstable
#   run converge -- it is exponentially sensitive to perturbation. So the
#   `stable` flag near the stability boundary depends on dt. Treat it as a
#   screening label, not ground truth, and do a dt-refinement check on a
#   handful of near-boundary cases.
#
# * The `stable` label here is a first-swing angle-separation screen. It is not
#   center-of-inertia referenced and says nothing about longer horizons.
#
# ===========================================================================

using GradPower
using Printf
using Random

include(joinpath(@__DIR__, "dynstab_io.jl"))

const E = joinpath(@__DIR__, "..", "examples")

# ---------------------------------------------------------------------------
# 0. Pick a case.
# ---------------------------------------------------------------------------
# Start small. IEEE9 has 3 machines and runs in milliseconds, so you can
# iterate. Switch to ACTIVSg2000 only once the sweep does what you expect.
#
#   ACTIVSg2000: raw/dyr below, and set add_static_gen_stubs => true.
#   Be aware it is only PARTIALLY modelled -- see the note at the bottom.

case = Dict(
    "raw" => joinpath(E, "ieee9_v33.raw"),
    "dyr" => joinpath(E, "ieee9bus_gov.dyr"),
    "add_static_gen_stubs" => true,
)

# ZIP load model exponent. alpha = 1.0 is constant power (the parser default);
# alpha = 0.5 splits constant-power / constant-impedance and is what the
# repository's reference comparisons use.
zip_alpha = 0.5

# Integration. dt = 1/120 s with backward Euler is the repo default.
dt      = 1 / 120
t_final = 3.0

# ---------------------------------------------------------------------------
# 1. Build the system once.
# ---------------------------------------------------------------------------
# from_psse parses the .raw (network + power flow data) and the .dyr (dynamic
# device models), build_network! forms Ybus, runpf! solves the base power flow.
# `build_system` does all three.

@info "Building case..."
sys = build_system(case)
@printf("  %d buses, %d branches, %d dynamic devices (diff=%d, alg=%d)\n",
        length(sys.buses), length(sys.branches), sys.dynamic.num_devices,
        sys.dynamic.diff_dim, sys.dynamic.alg_dim)

# Snapshot the lambda = 1 operating point. Every lambda is applied to THIS,
# so repeated scaling never compounds.
base = capture_base(sys)

# ---------------------------------------------------------------------------
# 2. Define the three input dimensions.
# ---------------------------------------------------------------------------

# (1) LOADING: sample lambda ~ Uniform(low, high). The base case is the mean
#     when low + high == 2. Seed it so the dataset is reproducible.
rng        = MersenneTwister(20240917)
n_lambda   = 3
lambda_lo, lambda_hi = 0.90, 1.10
lambdas    = [lambda_lo + (lambda_hi - lambda_lo) * rand(rng) for _ in 1:n_lambda]

# (2) FAULT LOCATION: external PSS/E bus numbers. `sys.busmap` maps those to
#     the internal indices the solver wants.
fault_buses_ext = [5, 7]

# (3) FAULT ADMITTANCE: shunt resistance, y = 1/r. See the non-monotonicity
#     warning above before choosing a range.
r_faults = [0.01, 0.1]

t_on, duration = 0.2, 0.1     # fault applied at 0.2 s, cleared 0.1 s later

@printf("\nSweep: %d lambda x %d bus x %d r_fault = %d scenarios\n",
        n_lambda, length(fault_buses_ext), length(r_faults),
        n_lambda * length(fault_buses_ext) * length(r_faults))
@printf("lambda draws: %s\n\n", join((@sprintf("%.4f", l) for l in lambdas), ", "))

# ---------------------------------------------------------------------------
# 3. Run the sweep.
# ---------------------------------------------------------------------------
# Loop order matters for cost. Changing lambda means re-solving the power flow
# and re-initializing the dynamics; changing the fault does not. So lambda is
# the OUTER loop and its setup cost is amortized over every fault beneath it.

results = NamedTuple[]

for (li, lambda) in enumerate(lambdas)

    # --- new operating point -------------------------------------------------
    # Scale loads and non-slack generation, re-solve the power flow, and
    # re-sync the ZIP load devices (including their reference voltage v0mag,
    # which must follow the new power-flow solution -- see apply_load_scale!).
    apply_load_scale!(sys, base, lambda; zip_alpha=zip_alpha)

    # Initialize the dynamic states from the power flow. This ASSERTS that the
    # residual is ~0, i.e. that we really are sitting at an equilibrium. If the
    # operating point were inconsistent, it would throw here rather than
    # silently produce a trajectory that drifts.
    empty!(sys.dynamic.events)
    dprob, residual = initialized_problem(sys)
    z0 = copy(dprob.zvec)          # every fault below restarts from this
    @printf("lambda = %.4f  (equilibrium residual %.2e)\n", lambda, residual)

    for bus_ext in fault_buses_ext, r_fault in r_faults

        bus_internal = sys.busmap[bus_ext]

        # --- apply the fault and integrate ---------------------------------
        empty!(sys.dynamic.events)          # clear the previous scenario
        dprob.zvec .= z0                    # rewind to the operating point
        GradPower.add_event!(sys,
            GradPower.ContingencyEvent(bus_internal, r_fault, t_on, t_on + duration))

        tvec, traj = GradPower.integrate!(dprob, sys, t_final; dt=dt)

        # --- extract channels and label ------------------------------------
        # traj is [n_states, n_steps+1]. dynamics_channels pulls out the parts
        # you actually want as [n, T] Float32: bus voltage magnitude/angle and
        # per-machine rotor angle / speed deviation.
        ch = dynamics_channels(sys, traj; downsample=1)
        m  = stability_metrics(sys, traj;
                               angle_sep_threshold=deg2rad(180.0), settle_ratio=0.5)

        push!(results, (lambda=lambda, bus=bus_ext, r_fault=r_fault,
                        sep=m["max_angle_sep_deg"], fdev=m["max_freq_dev"],
                        stable=m["stable"]))

        @printf("   bus %-3d r=%-5.3f -> max sep %7.2f deg | max |dw| %.3e | %s\n",
                bus_ext, r_fault, m["max_angle_sep_deg"], m["max_freq_dev"],
                m["stable"] ? "STABLE" : "UNSTABLE")
    end
    println()
end

# ---------------------------------------------------------------------------
# 4. What you just produced.
# ---------------------------------------------------------------------------
nstable = count(r -> r.stable, results)
@printf("%d scenarios: %d stable, %d unstable\n", length(results), nstable,
        length(results) - nstable)

println("""

Shapes per scenario (for the case above):
  bus_vm, bus_va      [n_bus, T]   voltage magnitude (pu) and angle (rad)
  gen_delta, gen_omega[n_gen, T]   rotor angle (rad), speed deviation (pu)

To generate a real dataset, do NOT extend this script. Use the production
driver, which shards the same sweep across MPI ranks and writes compressed
HDF5:

  julia --project=scripts scripts/generate_dynamics.jl scripts/sweeps/activs2000.toml --dry-run
  julia --project=scripts scripts/generate_dynamics.jl scripts/sweeps/activs2000.toml

and on Polaris:

  SWEEP=scripts/sweeps/activs2000.toml qsub -V -q debug -l select=1 \\
      -l walltime=01:00:00 scripts/hpc/gen-polaris.sh

Always run --dry-run first: it prints the scenario count and the sampled
lambda range without simulating anything.

CAVEAT on ACTIVSg2000 -- do not generate data from it yet:

1. THE CASE IS FINE. GRADPOWER'S POWER FLOW IS NOT (yet). runpf! does not
   enforce generator reactive-power limits, so 200 of 432 generators solve
   outside their Q limits -- bus 7400 absorbs 279 MVAr against an 8 MVAr
   floor. That drags 20 machines past their pull-out angle (internal angle
   > 90 deg, worst 164 deg), where synchronizing torque is negative, giving
   18 unstable eigenvalues (max Re +10.06). The trajectory then runs away
   regardless of the fault, so labels carry no information.

   uqgrid on the same case, same models, same fault:
       enforce_q_limits = False  ->  209.981 deg,  |w| 5.12e-2
       enforce_q_limits = True   ->    0.001 deg,  |w| 3.67e-7

   The fix is PV->PQ switching in the power flow (plan_enhance.md Phase 6).
   Use ACTIVSg200 or IEEE39 meanwhile: both have zero unstable eigenvalues.

2. MODEL COVERAGE. Separately, 858 .dyr records have no native kernel and are
   silently skipped unless you pass surrogates => true, which maps them onto
   TGOV1/SEXS in two tiers (440 mirrored from uqgrid, 418 GradPower-local with
   no oracle). Surrogates are stand-ins, never native coverage. Note this is
   NOT what causes the instability -- removing exciters entirely changes
   nothing.

   Full analysis: docs/activsg2000-diagnosis.md
""")
