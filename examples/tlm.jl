using GradPower
using SparseArrays
using FiniteDiff
using LinearAlgebra

raw_file = "examples/2bus.raw"
dyr_file = "examples/2bus.dyr"

# parse
raw = GradPower.read_psse_raw(raw_file)
sys = GradPower.raw_to_grad(raw)
psd = GradPower.PowerSystemDynamics(dyr_file)
GradPower.set_dynamics!(sys, psd)

# power flow
GradPower.build_network!(sys)
GradPower.runpf!(sys, verbose=false);

# dynamic simulation
tfinal = 1.0
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

# integrate dynamics
event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))
tvec, traj = GradPower.integrate!(dprob, sys, tfinal)

# compute TLM w.r.t. initial condition
δz0 = ones(length(dprob.zvec))

δztf = GradPower.tlm(δz0, dprob, sys, traj, tvec)

# compute TLM w.r.t initial condition using finite differences
function final_state(ϵ)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    dprob.zvec .+= ϵ*ones(length(dprob.zvec))
    tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
    return traj[:, end]
end

ϵ = 1e-6
traj1 = final_state(0)
traj2 = final_state(ϵ)
traj3 = final_state(-ϵ)

δztf_fd = (traj2 - traj3) / (2*ϵ)
error = norm(δztf - δztf_fd, Inf)

println("TLM initial conditions error = $error")

# compute TLM w.r.t. parameters
δz0 = zeros(length(dprob.zvec))
δp = ones(length(dprob.pvec)) # direction of perturbation

δztf = GradPower.tlm(δz0, dprob, sys, traj, tvec, δp=δp, finite_diff=false)

function final_state_param(p)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    dprob.pvec .+= p*ones(length(dprob.pvec))
    tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
    return traj[:, end]
end

ϵ = 1e-4
traj1 = final_state_param(0)
traj2 = final_state_param(ϵ)
traj3 = final_state_param(-ϵ)

δztf_fd = (traj2 - traj3) / (2*ϵ)

error = norm(δztf - δztf_fd, Inf)
println("TLM parameter error = $error")
