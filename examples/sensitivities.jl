using Plots
using Revise    
using GradPower
using SparseArrays
using FiniteDiff
using ForwardDiff
using LinearAlgebra

using Profile
using PProf

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
tfinal = 1.0
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

# integrate dynamics
#event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))
tvec, traj = GradPower.integrate!(dprob, sys, tfinal)

# compute TLM w.r.t. initial condition
state_idx = 3
δz0 = zeros(length(dprob.zvec))
δz0[state_idx] = 1.0
δz0 = ones(length(dprob.zvec))

@time δztf = GradPower.tlm(δz0, dprob, sys, traj, tvec)

# compute TLM w.r.t initial condition using finite differences
function final_state(ϵ)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    #dprob.zvec[state_idx] += ϵ
    dprob.zvec .+= ϵ*ones(length(dprob.zvec))
    tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
    return traj[:, end]
end

ϵ = 1e-6
traj1 = final_state(0)
traj2 = final_state(ϵ)
traj3 = final_state(-ϵ)

δztf_fd = (traj2 - traj3) / (2*ϵ)
#println("δztf_fd = $δztf_fd")
#println("δztf = $δztf")

error = norm(δztf - δztf_fd, Inf)
println("TLM initial conditions error = $error")

# compute TLM w.r.t. parameters
δz0 = zeros(length(dprob.zvec))
δp = zeros(length(dprob.pvec)) # direction of perturbation
pidx = 20
δp[pidx] = 1.0

GradPower.initialize_dynamics!(dprob, sys)
#event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))
tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
@time δztf = GradPower.tlm(δz0, dprob, sys, traj, tvec, δp=δp, finite_diff=true)

function final_state_param(p)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    dprob.pvec[pidx] += p
    tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
    return traj[:, end]
end

ϵ = 1e-4
traj1 = final_state_param(0)
traj2 = final_state_param(ϵ)
traj3 = final_state_param(-ϵ)

δztf_fd = (traj2 - traj3) / (2*ϵ)
#println("δztf_fd = $δztf_fd")

error = norm(δztf - δztf_fd, Inf)
println("TLM parameter error = $error")
