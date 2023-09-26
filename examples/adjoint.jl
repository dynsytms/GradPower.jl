using Revise    
using GradPower
using SparseArrays
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using UnicodePlots

raw_file = "examples/2bus.raw"
dyr_file = "examples/2bus.dyr"

#raw_file = "examples/ieee9_v33.raw"
#dyr_file = "examples/ieee9bus.dyr"

#raw_file = "examples/ACTIVSg2000.raw"
#dyr_file = "examples/ACTIVSg2000.dyr"

# parse
raw = GradPower.read_psse_raw(raw_file)
sys = GradPower.raw_to_grad(raw)
psd = GradPower.PowerSystemDynamics(dyr_file)
GradPower.set_dynamics!(sys, psd)

# power flow
GradPower.build_network!(sys)
GradPower.runpf!(sys, verbose=false);

# dynamic simulation
dt = 1.0/120.0
tfinal = 100.0/120.0
#tfinal = 1.0
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

# integrate dynamics
#event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))
tvec, traj = GradPower.integrate!(dprob, sys, tfinal)

state_idx = 4
λ0 = zeros(length(dprob.zvec))
λ0[state_idx] = 1.0

λ, μ = GradPower.adjoint(λ0, dprob, sys, traj, tvec)


# compute finite differences.

function final_state(x)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    dprob.zvec .= x
    tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
    return traj[state_idx, end]
end

function final_state_p(p)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    dprob.pvec .= p
    tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
    return traj[state_idx, end]
end

znom = dprob.zvec
λfd = FiniteDiff.finite_difference_gradient(final_state, znom)
pnom = dprob.pvec
μfd = FiniteDiff.finite_difference_gradient(final_state_p, pnom)

println("Compute sensitivities w.r.t final state")
println("Compute λ")
println(λ)
println(λfd)
println("Compute μ")
println(μ)
println(μfd)

println("")
println("")

# now, include functional

function objective_numeric(tvec, traj, u, p, sys)
    val = 0.0
    for i=1:(size(traj, 2) - 1)
        rfun = GradPower.functional(traj[:, i + 1], u, p, sys)
        val += (tvec[i + 1] - tvec[i])*rfun
    end
    return val
end

function functional_state(x)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    dprob.zvec .= x
    tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
    return objective_numeric(tvec, traj, dprob.uvec, dprob.pvec, sys)
end
eps = 1e-6
zz = copy(znom)
zz[5] += eps
obj = functional_state(zz)
fd_obj = FiniteDiff.finite_difference_gradient(functional_state, znom)
println(obj/eps)

λ0 = zeros(length(dprob.zvec))
λfun, μfun = GradPower.adjoint(λ0, dprob, sys, traj, tvec, functional=true)

println("Compute sensitivities w.r.t. functional")
println("Compute λ")
println(λfun)
println(fd_obj)
println("Compute μ")
println(μfun)

println("")
println("")
println("Compute sensitivities w.r.t. functional")
