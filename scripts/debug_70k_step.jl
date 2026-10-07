using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GradPower

REPO = abspath(joinpath(@__DIR__, ".."))

# Check what the phase 6 baseline had for activs70k
# Phase 6: newton=799, residual=1041, nsteps=240 -> ~3.3 newton/step
# Phase 10: newton=3951, residual=4125, nsteps=240 -> ~16.5 newton/step

# Let's check: does the solver hit max_newton on some steps?
# First let's look at integrate! to understand max_newton

ps = GradPower.from_psse(
    joinpath(REPO, "examples", "ACTIVSg70k.raw"),
    joinpath(REPO, "examples", "ACTIVSg70k.dyr"))
GradPower.build_network!(ps)
GradPower.runpf!(ps)
for dev in ps.dynamic.devices
    if dev.dtype isa GradPower.ZIPLoad
        dev.dtype.α = 0.5
    end
end

n_ieeest = count(d -> d.dtype isa GradPower.IEEEST, ps.dynamic.devices)
n_esdc1a = count(d -> d.dtype isa GradPower.ESDC1A, ps.dynamic.devices)
n_genrou = count(d -> d.dtype isa GradPower.Genrou, ps.dynamic.devices)
n_static = count(d -> d.dtype isa GradPower.StaticGenerator, ps.dynamic.devices)
println("Genrou: $n_genrou, ESDC1A: $n_esdc1a, IEEEST: $n_ieeest, StaticGen: $n_static")
println("diff_dim=$(ps.dynamic.diff_dim), alg_dim=$(ps.dynamic.alg_dim)")

# The phase 6 baseline had sys_dim=214194. Current has 216698.
# diff: 216698 - 214194 = 2504 extra states.
# ESDC1A: diff_size=3, alg_size=1 -> 4 states per device
# IEEEST: diff_size=?, alg_size=?
println("\n--- Checking IEEEST/ESDC1A state sizes ---")
for dev in ps.dynamic.devices
    if dev.dtype isa GradPower.IEEEST
        println("IEEEST: diff_size=$(dev.dtype.diff_size), alg_size=$(dev.dtype.alg_size)")
        break
    end
end
for dev in ps.dynamic.devices
    if dev.dtype isa GradPower.ESDC1A
        println("ESDC1A: diff_size=$(dev.dtype.diff_size), alg_size=$(dev.dtype.alg_size)")
        break
    end
end
# 313 ESDC1A * (3+1) + 313 IEEEST * ? = 2504
# 313 * 4 = 1252; remainder = 1252 -> IEEEST has 4 states each? 1252/313 = 4

dp = GradPower.DynamicProblem(ps)
GradPower.initialize_dynamics!(dp, ps)
z0 = copy(dp.zvec)

# Short run: just past the fault application
fault_bus_int = ps.busmap[3]
GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, 0.02, 0.2, 0.3))

# Run only 10 steps
println("\n--- Short integration (10 steps, ~0.083s) ---")
slog = GradPower.SolverLog()
t0 = time()
GradPower.integrate!(dp, ps, 10.0/120.0; dt=1.0/120.0, log=slog)
elapsed = time() - t0
println("$(round(elapsed, digits=1))s, newton=$(slog.jacobian_count), residual=$(slog.residual_count)")
println("newton/step: $(round(slog.jacobian_count / 10.0, digits=1))")

# Run more steps to get past the fault
println("\n--- Continuing to t=0.5s (past fault) ---")
dp.zvec .= z0
empty!(ps.dynamic.events)
GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, 0.02, 0.2, 0.3))
slog2 = GradPower.SolverLog()
t0 = time()
GradPower.integrate!(dp, ps, 0.5; dt=1.0/120.0, log=slog2)
elapsed = time() - t0
println("$(round(elapsed, digits=1))s, newton=$(slog2.jacobian_count), residual=$(slog2.residual_count)")
nsteps = Int(round(0.5 / (1.0/120.0)))
println("newton/step: $(round(slog2.jacobian_count / nsteps, digits=1))")
