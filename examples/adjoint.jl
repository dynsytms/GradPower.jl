using Revise    
using GradPower
using SparseArrays
using FiniteDiff
using ForwardDiff
using LinearAlgebra


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
tfinal = 2/120.0
#tfinal = 1.0
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

# integrate dynamics
#event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))
tvec, traj = GradPower.integrate!(dprob, sys, tfinal)

state_idx = 3
λ0 = zeros(length(dprob.zvec))
λ0[state_idx] = 1.0

@time λ = GradPower.adjoint(λ0, dprob, sys, traj, tvec)

# compute TLM w.r.t initial condition using finite differences
function final_state(ϵ)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    dprob.zvec[state_idx] += ϵ
    #dprob.zvec .+= ϵ*ones(length(dprob.zvec))
    tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
    return traj[:, end]
end

ϵ = 1e-6
traj1 = final_state(0)
traj2 = final_state(ϵ)
traj3 = final_state(-ϵ)

λfd = (traj2 - traj3) / (2*ϵ)

error = norm(λ - λfd, Inf)
println("TLM initial conditions error = $error")
