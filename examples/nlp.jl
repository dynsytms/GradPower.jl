using Revise
using GradPower
using MadNLP
using NLPModels
using NPZ

raw_file = "examples/2bus.raw"
dyr_file = "examples/2bus.dyr"

raw_file = "examples/ieee9_v33.raw"
dyr_file = "examples/ieee9bus.dyr"

raw_file = "examples/ACTIVSg200.raw"
dyr_file = "examples/ACTIVSg200.dyr"

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
tfinal = 2.0
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

# integrate dynamics
event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))
tvec, traj = GradPower.integrate!(dprob, sys, tfinal)

# save nominal trajectory
npzwrite("traj.npz", Dict("tvec"=>tvec, "traj"=>traj))

# compute adjoint
λ0 = zeros(length(dprob.zvec))
λfun, μfun = GradPower.adjoint(λ0, dprob, sys, traj, tvec, functional=true)

# save gradient of functional with respect to parameters
npzwrite("grad.npz", Dict("grad"=>μfun))

# build nlp
lvar = copy(dprob.pvec)
uvar = copy(dprob.pvec)


# find number of dynamic generators
ngen = 0
for device in sys.dynamic.devices
    if device.dtype isa Genrou
        global ngen += 1
    end
end

for i in 1:ngen
    # adjust inertia
    lvar[7 + 12*(i-1)] = 0.8*lvar[7 + 12*(i-1)]
    uvar[7 + 12*(i-1)] = 1.2*uvar[7 + 12*(i-1)]
end


nlp = GradPower.DynamicNLP(sys, dprob, tfinal, lvar, uvar)

# solve problem
solver = MadNLP.MadNLPSolver(nlp;
    print_level=MadNLP.INFO,
    kkt_system=MadNLP.DENSE_KKT_SYSTEM,
    hessian_approximation=MadNLP.DENSE_BFGS,
    linear_solver=LapackCPUSolver,
)
MadNLP.solve!(solver)

# obtain optimal pvec
pvec_opt = solver.x.values

# integrate dynamics
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)
dprob.pvec .= pvec_opt
tvec, traj = GradPower.integrate!(dprob, sys, tfinal)

# save optimal trajectory
npzwrite("traj_opt.npz", Dict("tvec"=>tvec, "traj"=>traj))
