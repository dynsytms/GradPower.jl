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

@time λ = GradPower.adjoint(λ0, dprob, sys, traj, tvec)

function final_state(x)
    dprob = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dprob, sys)
    dprob.zvec .= x
    tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
    return traj[state_idx, end]
end

if false
J = GradPower.preallocate_jacobian(sys)
GradPower.rhs_jac!(J, traj[:, end], dprob.uvec, dprob.pvec, sys)
Jd = Array(J)
Jt = transpose(Jd)
diff_dim = sys.dynamic.diff_dim
Jt[1:diff_dim, :] .*= dt
for j = 1:diff_dim
    Jt[j, j] -= 1.0
end
rhs = copy(λ0)
rhs *= -1
λs = Jt \ rhs
end

# compute other stuff
gg = FiniteDiff.finite_difference_gradient(final_state, znom)
